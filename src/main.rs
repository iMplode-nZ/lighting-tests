use std::collections::HashSet;
use std::env::current_exe;

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

    let walls = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn(Tex2d<Vec4<f32>>)>::new_async(
        &device,
        &track!(|light| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos >> SCALE_POWER;
            let power = light.read(pos).reduce_sum();
            let color = power
                * if (emission.read(pos) == 0.0).all() {
                    Vec3::splat_expr(1.0)
                } else {
                    Vec3::new(1.0, 0.0, 0.0).expr()
                };
            let color = color.var();
            if walls.read(pos) != 0 {
                *color = Vec3::new(0.0, 1.0, 0.0).expr();
            }
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

    let update_wall_kernel = Kernel::<fn(Vec2<u32>, u32)>::new_async(
        &device,
        &track!(|pos, wall| {
            walls.write(pos, wall);
        }),
    );

    let update_kernel = Kernel::<fn(Tex2d<Vec4<f32>>, Tex2d<Vec4<f32>>)>::new_async(
        &device,
        &track!(|lights, next_lights| {
            let x = Vec2::new(1_u32, 0);
            let y = Vec2::new(0, 1_u32);
            let pos = dispatch_id().xy() + 1;

            let power = 0.0.var();
            let light = Vec4::<f32>::var_zeroed();

            let l = lights.read(pos - x);
            if l.reduce_sum() > EPSILON {
                let p = l.z;
                *light += l * p / l.reduce_sum();
                *power += p;
            }

            let l = lights.read(pos - y);
            if l.reduce_sum() > EPSILON {
                let p = l.w;
                *light += l * p / l.reduce_sum();
                *power += p;
            }

            let l = lights.read(pos + x);
            if l.reduce_sum() > EPSILON {
                let p = l.x;
                *light += l * p / l.reduce_sum();
                *power += p;
            }

            let l = lights.read(pos + y);
            if l.reduce_sum() > EPSILON {
                let p = l.y;
                *light += l * p / l.reduce_sum();
                *power += p;
            }
            if light.reduce_sum() <= EPSILON {
                return;
            }

            if (walls.read(pos - x) & WALL_REFLECT) != 0 {
                *light = Vec4::new(0.0, 1.0, 1.0, 1.0);
            }
            if (walls.read(pos - y) & WALL_REFLECT) != 0 {
                *light = Vec4::new(1.0, 0.0, 1.0, 1.0);
            }
            if (walls.read(pos + x) & WALL_REFLECT) != 0 {
                *light = Vec4::new(1.0, 1.0, 0.0, 1.0);
            }
            if (walls.read(pos + y) & WALL_REFLECT) != 0 {
                *light = Vec4::new(1.0, 1.0, 1.0, 0.0);
            }
            let light = light / light.reduce_sum() * power;

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
            update_emission_kernel.dispatch([1, 1, 1], &pos, &Vec4::new(0.7, 0.1, 0.1, 0.1));
        }
        if active_buttons.contains(&MouseButton::Right) {
            update_wall_kernel.dispatch([1, 1, 1], &pos, &WALL_REFLECT);
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
