use std::collections::HashSet;
use std::env::current_exe;
use std::thread::sleep;
use std::time::Duration;

use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, Event, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use luisa::lang::types::vector::*;
use luisa::prelude::*;
use luisa_compute as luisa;

const GRID_SIZE: u32 = 128;
const SCALING: u32 = 8;
const SCALE_POWER: u32 = 3;

const COLOR_RANGE: f32 = 1.0;

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

    let c: f32 = 1.0;
    let dt: f32 = 1.0 / 60.0;
    let dx: f32 = 1.0;

    let velocity = device.create_tex2d::<f32>(PixelStorage::Float1, GRID_SIZE, GRID_SIZE, 1);
    let height = device.create_tex2d::<f32>(PixelStorage::Float1, GRID_SIZE, GRID_SIZE, 1);

    let walls = device.create_tex2d::<u32>(PixelStorage::Byte1, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn()>::new_async(
        &device,
        &track!(|| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos >> SCALE_POWER;
            let color = Vec3::splat(1.0) * (height.read(pos) / COLOR_RANGE * 0.5 + 0.5);
            let color = color.var();

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

    let set_height_kernel = Kernel::<fn(Vec2<u32>, f32)>::new_async(
        &device,
        &track!(|pos, value| {
            height.write(pos, value);
        }),
    );

    let update_wall_kernel = Kernel::<fn(Vec2<u32>, u32)>::new_async(
        &device,
        &track!(|pos, wall| {
            walls.write(pos, wall);
        }),
    );

    let timestep_kernel = Kernel::<fn()>::new_async(
        &device,
        &track!(|| {
            let pos = dispatch_id().xy();
            let v = velocity.read(pos);
            let h = height.read(pos);
            if walls.read(pos) == WALL_ABSORB {
                height.write(pos, 0.0);
            } else {
                height.write(pos, h + v * dt);
            }
        }),
    );

    let update_kernel = Kernel::<fn()>::new_async(
        &device,
        &track!(|| {
            let x = Vec2::new(1_u32, 0);
            let y = Vec2::new(0, 1_u32);
            let pos = dispatch_id().xy() + 1;
            let h = height.read(pos);
            let ddx = (height.read(pos + x) - 2.0 * h + height.read(pos - x)) / (dx * dx);
            let ddy = (height.read(pos + y) - 2.0 * h + height.read(pos - y)) / (dx * dx);
            let a = c * c * (ddx + ddy);
            velocity.write(pos, velocity.read(pos) + a * dt);
        }),
    );

    let mut cursor_pos = PhysicalPosition::new(0.0, 0.0);

    let mut active_buttons = HashSet::new();

    let mut t = 0.0_f32;

    let update_cursor = |active_buttons: &HashSet<MouseButton>,
                         cursor_pos: PhysicalPosition<f64>| {
        let pos = Vec2::new(
            (cursor_pos.x as u32) >> SCALE_POWER,
            (cursor_pos.y as u32) >> SCALE_POWER,
        );
        if active_buttons.contains(&MouseButton::Right) {
            update_wall_kernel.dispatch([1, 1, 1], &pos, &WALL_ABSORB);
        }
        if active_buttons.contains(&MouseButton::Middle) {
            update_wall_kernel.dispatch([1, 1, 1], &pos, &WALL_DIFFUSE);
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
                    {
                        let scope = device.default_stream().scope();
                        scope.present(&swapchain, &display);
                        let commands = vec![
                            set_height_kernel.dispatch_async([1, 1, 1], &Vec2::splat(64), &t.sin()),
                            update_kernel.dispatch_async([GRID_SIZE - 2, GRID_SIZE - 2, 1]),
                            timestep_kernel.dispatch_async([GRID_SIZE, GRID_SIZE, 1]),
                            draw_kernel.dispatch_async([
                                GRID_SIZE * SCALING,
                                GRID_SIZE * SCALING,
                                1,
                            ]),
                        ];
                        scope.submit(commands);
                    }
                    window.request_redraw();
                    t += dt;
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
