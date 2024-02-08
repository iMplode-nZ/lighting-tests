use std::collections::HashSet;
use std::env::current_exe;
use std::f32::consts::PI;

use glam::IVec2;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use luisa::lang::types::vector::*;
use luisa::prelude::*;
use luisa_compute as luisa;

use crate::light::{compute_gathers_2, compute_simple_gathers, compute_slope_gathers_n};

mod light;

const GRID_SIZE: u32 = 128;
const SCALING: u32 = 8;
const SCALE_POWER: u32 = 3;

const DIRECTIONS: u32 = 9;
const TOTAL_DIRECTIONS: u32 = DIRECTIONS * 4;

const STEP_DIFFUSE: f32 = 0.99;

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

    type Lights = Tex3d<Vec4<f32>>;

    let gathers = compute_slope_gathers_n(DIRECTIONS);

    let lights_a = device.create_tex3d::<Vec4<f32>>(
        PixelStorage::Float4,
        GRID_SIZE,
        GRID_SIZE,
        TOTAL_DIRECTIONS,
        1,
    );
    let lights_b = device.create_tex3d::<Vec4<f32>>(
        PixelStorage::Float4,
        GRID_SIZE,
        GRID_SIZE,
        TOTAL_DIRECTIONS,
        1,
    );

    let emission = device.create_tex3d::<Vec4<f32>>(
        PixelStorage::Float4,
        GRID_SIZE,
        GRID_SIZE,
        TOTAL_DIRECTIONS,
        1,
    );

    let walls = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn(Lights)>::new(
        &device,
        &track!(|lights| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos >> SCALE_POWER;
            let color = Vec3::<f32>::var_zeroed();
            for i in 0..TOTAL_DIRECTIONS {
                *color += lights.read(pos.extend(i)).xyz();
            }

            // TODO: Tonemap

            let w = walls.read(pos);
            if w == WALL_ABSORB {
                *color = Vec3::new(1.0, 0.0, 0.0).expr();
            } else if w == WALL_DIFFUSE {
                *color = Vec3::new(0.0, 1.0, 0.0).expr();
            } else if w == WALL_REFLECT {
                *color = Vec3::new(0.0, 0.0, 1.0).expr();
            }

            display.write(display_pos, color.extend(1.0));
        }),
    );

    let emit_kernel = Kernel::<fn(Lights)>::new(
        &device,
        &track!(|lights| {
            let pos = dispatch_id();
            let emission = emission.read(pos);
            if (emission != 0.0).any() {
                lights.write(pos, emission);
            }
        }),
    );

    let update_emission_kernel =
        Kernel::<fn(Vec2<u32>, [Vec3<f32>; TOTAL_DIRECTIONS as usize])>::new(
            &device,
            &track!(|pos, light| {
                emission.write(
                    pos.extend(dispatch_id().z),
                    light[dispatch_id().z].extend(0.0),
                );
            }),
        );

    let update_wall_kernel = Kernel::<fn(Vec2<u32>, u32)>::new(
        &device,
        &track!(|pos, wall| {
            walls.write(pos, wall);
        }),
    );

    let update_kernel = Kernel::<fn(Lights, Lights)>::new(
        &device,
        &track!(|lights, next_lights| {
            let pos = dispatch_id().xy() + 1;

            let light = [Vec4::<f32>::splat(0.0); TOTAL_DIRECTIONS as usize].var();

            fn rotate(face: u32, dir: IVec2) -> IVec2 {
                // TODO: Make sure these are right; it doesn't really matter though.
                match face {
                    // +x
                    0 => dir,
                    // -y
                    1 => IVec2::new(-dir.y, dir.x),
                    // -x
                    2 => -dir,
                    // y
                    3 => IVec2::new(dir.y, -dir.x),
                    _ => panic!("Invalid"),
                }
            }

            let apply = |face: u32| {
                escape!({
                    for gather in &gathers {
                        track!({
                            let dir = gather.direction + face * DIRECTIONS;
                            let transmission = gather.quantity
                                * lights.read(
                                    (pos.cast_i32()
                                        + Vec2::<i32>::from(rotate(face, gather.offset)))
                                    .cast_u32()
                                    .extend(dir),
                                );
                            light.write(dir, transmission + light[dir]);
                        })
                    }
                })
            };

            apply(0);
            apply(1);
            apply(2);
            apply(3);

            for dir in 0..TOTAL_DIRECTIONS {
                light.write(dir, light[dir] * STEP_DIFFUSE);
            }

            let w = walls.read(pos);
            if w == WALL_ABSORB {
                *light = [Vec4::splat(0.0); TOTAL_DIRECTIONS as usize];
            }

            if w == WALL_DIFFUSE {
                let total_light = Vec4::<f32>::var_zeroed();
                for i in 0..TOTAL_DIRECTIONS {
                    *total_light += light[i];
                }
                *total_light /= TOTAL_DIRECTIONS as f32;
                for i in 0..TOTAL_DIRECTIONS {
                    *light[i] = total_light;
                }
            }

            for i in 0..TOTAL_DIRECTIONS {
                next_lights.write(pos.extend(i), light[i]);
            }
        }),
    );

    let mut parity = false;

    let mut cursor_pos = PhysicalPosition::new(0.0, 0.0);

    let mut active_buttons = HashSet::new();

    let light_color = Vec3::new(1.0, 0.7, 0.2);
    let light_magnitude = 15.0 / TOTAL_DIRECTIONS as f32;
    let light = Vec3::new(
        light_color.x * light_magnitude,
        light_color.y * light_magnitude,
        light_color.z * light_magnitude,
    );

    let total_light = [light; TOTAL_DIRECTIONS as usize];

    // total_light[4] = light;

    let update_cursor = |active_buttons: &HashSet<MouseButton>,
                         cursor_pos: PhysicalPosition<f64>| {
        let pos = Vec2::new(
            (cursor_pos.x as u32) >> SCALE_POWER,
            (cursor_pos.y as u32) >> SCALE_POWER,
        );
        if active_buttons.contains(&MouseButton::Left) {
            update_emission_kernel.dispatch([1, 1, TOTAL_DIRECTIONS], &pos, &total_light);
        }
        if active_buttons.contains(&MouseButton::Right) {
            update_wall_kernel.dispatch([1, 1, 1], &pos, &WALL_ABSORB);
        }
        if active_buttons.contains(&MouseButton::Middle) {
            update_wall_kernel.dispatch([1, 1, 1], &pos, &WALL_DIFFUSE);
        }
    };
    let update_cursor = &update_cursor;

    println!("Ready");

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
                            emit_kernel.dispatch_async(
                                [GRID_SIZE, GRID_SIZE, TOTAL_DIRECTIONS],
                                next_lights,
                            ),
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
