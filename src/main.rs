use std::env::current_exe;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use luisa::lang::types::vector::*;
use luisa::prelude::*;
use luisa_compute as luisa;

const GRID_SIZE: u32 = 128;
const GRID_POWER: u32 = 7;
const SCALING: u32 = 8;
const SCALE_POWER: u32 = 3;
const MAX_LIGHT: u32 = 16;

#[tracked]
fn hash(x: Expr<u32>) -> Expr<u32> {
    let x = x.var();
    *x ^= x >> 17;
    *x *= 0xed5ad4bb;
    *x ^= x >> 11;
    *x *= 0xac4c1b51;
    *x ^= x >> 15;
    *x *= 0x31848bab;
    *x ^= x >> 14;
    **x
}

#[tracked]
fn rand_at(pos: Expr<Vec2<u32>>, t: Expr<u32>) -> Expr<u32> {
    let input = pos.x + pos.y << GRID_POWER + t << (GRID_POWER * 2);
    hash(input)
}

fn main() {
    luisa::init_logger();
    let ctx = Context::new(current_exe().unwrap());
    let device = ctx.create_device("cuda");

    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
        .with_inner_size(winit::dpi::PhysicalSize::new(
            GRID_SIZE * SCALING,
            GRID_SIZE * SCALING,
        ))
        .with_resizable(false)
        .build(&event_loop)
        .unwrap();

    let swapchain = device.create_swapchain(
        &window,
        &device.default_stream(),
        GRID_SIZE * SCALING,
        GRID_SIZE * SCALING,
        false,
        false,
        3,
    );
    let display = device.create_tex2d::<Vec4<f32>>(
        swapchain.pixel_storage(),
        GRID_SIZE * SCALING,
        GRID_SIZE * SCALING,
        1,
    );

    // Directions: (-1, 0) - x, (0, -1) - y, (1, 0) - z, (0, 1) - w
    let lights_a = device.create_tex2d::<Vec4<u32>>(PixelStorage::Byte4, GRID_SIZE, GRID_SIZE, 1);
    let lights_b = device.create_tex2d::<Vec4<u32>>(PixelStorage::Byte4, GRID_SIZE, GRID_SIZE, 1);

    let emission = device.create_tex2d::<Vec4<u32>>(PixelStorage::Byte4, GRID_SIZE, GRID_SIZE, 1);
    let is_emitting = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn(Tex2d<Vec4<u32>>)>::new_async(
        &device,
        &track!(|lights| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos >> SCALE_POWER;
            let light = lights.read(pos);
            let color = light.reduce_sum().as_f32() / (4.0 * MAX_LIGHT as f32) * Vec3::splat(1.0);
            display.write(display_pos, color.extend(1.0));

            if is_emitting.read(pos) == 1 {
                display.write(display_pos, Vec4::new(1.0, 0.0, 0.0, 1.0));
            }
        }),
    );

    let emit_kernel = Kernel::<fn(Tex2d<Vec4<u32>>)>::new_async(
        &device,
        &track!(|lights| {
            let pos = dispatch_id().xy();
            if is_emitting.read(pos) == 1 {
                lights.write(pos, emission.read(pos));
            }
        }),
    );

    let update_emission_kernel = Kernel::<fn(Vec2<u32>, Vec4<u32>)>::new_async(
        &device,
        &track!(|pos, light| {
            is_emitting.write(pos, 1);
            emission.write(pos, light);
        }),
    );

    let update_kernel = Kernel::<fn(Tex2d<Vec4<u32>>, Tex2d<Vec4<u32>>)>::new_async(
        &device,
        &track!(|lights, next_lights| {
            let x = Vec2::new(1_u32, 0);
            let y = Vec2::new(0, 1_u32);
            let pos = dispatch_id().xy() + 1;
            let light = Vec4::<u32>::var_zeroed();
            *light.x = lights.read(pos + x).x;
            *light.y = lights.read(pos + y).y;
            *light.z = lights.read(pos - x).z;
            *light.w = lights.read(pos - y).w;

            let light = luisa::max(light, 1_u32) - 1;

            next_lights.write(pos, light);
        }),
    );

    let mut parity = false;

    update_emission_kernel.dispatch([1, 1, 1], &Vec2::splat(64), &Vec4::splat(MAX_LIGHT));

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id,
            } if window_id == window.id() => {
                elwt.exit();
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                window_id,
            } if window_id == window.id() => {
                let lights = if parity { &lights_a } else { &lights_b };
                let next_lights = if parity { &lights_b } else { &lights_a };
                parity = !parity;
                {
                    let scope = device.default_stream().scope();
                    scope.present(&swapchain, &display);
                    let commands = vec![
                        update_kernel.dispatch_async(
                            [GRID_SIZE - 2, GRID_SIZE - 2, 1],
                            lights,
                            next_lights,
                        ),
                        emit_kernel.dispatch_async([GRID_SIZE, GRID_SIZE, 1], next_lights),
                        draw_kernel.dispatch_async(
                            [GRID_SIZE * SCALING, GRID_SIZE * SCALING, 1],
                            next_lights,
                        ),
                    ];
                    scope.submit(commands);
                }
                window.request_redraw();
            }
            _ => (),
        })
        .unwrap();
}
