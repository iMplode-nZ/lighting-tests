use std::collections::HashSet;
use std::env::current_exe;
use std::f32::consts::PI;

use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use luisa::lang::types::vector::*;
use luisa::prelude::*;
use luisa_compute as luisa;

const GRID_SIZE: u32 = 128;
const SCALING: u32 = 8;
const SCALE_POWER: u32 = 3;
const EPSILON: f32 = 0.01;

const WALL_ABSORB: u32 = 0b01;
const WALL_REFLECT: u32 = 0b10;
const WALL_DIFFUSE: u32 = 0b100;

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
    let lights_a = device.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, GRID_SIZE, GRID_SIZE, 1);
    let lights_b = device.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, GRID_SIZE, GRID_SIZE, 1);

    let emission = device.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn(Tex2d<Vec4<f32>>)>::new_async(
        &device,
        &track!(|light| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos >> SCALE_POWER;
            let power = light.read(pos).reduce_sum();
            let color = power * Vec3::splat(1.0).expr();
            let color = color.var();

            display.write(display_pos, color.extend(1.0));
        }),
    );

    let emit_kernel = Kernel::<fn(Tex2d<Vec4<f32>>)>::new_async(
        &device,
        &track!(|lights| {
            let pos = dispatch_id().xy();
            let emission = emission.read(pos);
            if (emission != 0.0).any() {
                lights.write(pos, emission);
            }
        }),
    );

    let update_emission_kernel = Kernel::<fn(Vec2<u32>, Vec4<f32>)>::new_async(
        &device,
        &track!(|pos, light| {
            emission.write(pos, light);
        }),
    );

    let center_fraction = 1.0_f32.atan2(3.0) / (PI / 4.0);

    let update_kernel = Kernel::<fn(Tex2d<Vec4<f32>>, Tex2d<Vec4<f32>>)>::new_async(
        &device,
        &track!(|lights, next_lights| {
            let x = Vec2::new(1_i32, 0);
            let y = Vec2::new(0, 1_i32);
            let pos = dispatch_id().xy() + 1;

            fn mask(offset: Vec2<i32>) -> Vec4<f32> {
                if offset.x == -1 && offset.y == 0 {
                    Vec4::new(1.0, 0.0, 0.0, 0.0)
                } else if offset.x == 1 && offset.y == 0 {
                    Vec4::new(0.0, 0.0, 1.0, 0.0)
                } else if offset.x == 0 && offset.y == -1 {
                    Vec4::new(0.0, 1.0, 0.0, 0.0)
                } else if offset.x == 0 && offset.y == 1 {
                    Vec4::new(0.0, 0.0, 0.0, 1.0)
                } else {
                    panic!("Invalid offset");
                }
            }
            fn rev(offset: Vec2<i32>) -> Vec2<i32> {
                Vec2::new(-offset.x, -offset.y)
            }

            let light = Vec4::<f32>::var_zeroed();

            let apply = |offset: Vec2<i32>, normal: Vec2<i32>| {
                let l = lights.read((pos.cast_i32() + offset).cast_u32());
                let l = (l * mask(rev(offset))).reduce_sum() / 2.0;
                let c = 2.0 * center_fraction * l;
                let n = (1.0 - center_fraction) * l;
                *light += mask(rev(offset)) * c + mask(normal) * n + mask(rev(normal)) * n;
            };
            apply(x, y);
            apply(rev(x), y);
            apply(y, x);
            apply(rev(y), x);

            next_lights.write(pos, light);
        }),
    );

    let mut parity = false;

    let mut cursor_pos = PhysicalPosition::new(0.0, 0.0);

    let mut active_buttons = HashSet::new();

    let update_cursor = |active_buttons: &HashSet<MouseButton>,
                         cursor_pos: PhysicalPosition<f64>| {
        let pos = Vec2::new(
            (cursor_pos.x as u32) >> SCALE_POWER,
            (cursor_pos.y as u32) >> SCALE_POWER,
        );
        if active_buttons.contains(&MouseButton::Left) {
            update_emission_kernel.dispatch([1, 1, 1], &pos, &Vec4::new(1.0, 1.0, 1.0, 1.0));
        }
    };
    let update_cursor = &update_cursor;

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run(move |event, elwt| match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => {
                    elwt.exit();
                }
                WindowEvent::RedrawRequested => {
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
                WindowEvent::CursorMoved { position, .. } => {
                    cursor_pos = position;
                    update_cursor(&active_buttons, cursor_pos);
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    match state {
                        ElementState::Pressed => {
                            active_buttons.insert(button);
                        }
                        ElementState::Released => {
                            active_buttons.remove(&button);
                        }
                    }
                    update_cursor(&active_buttons, cursor_pos);
                }
                _ => (),
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => (),
        })
        .unwrap();
}
