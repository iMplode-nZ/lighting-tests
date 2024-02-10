use std::collections::HashSet;
use std::env::current_exe;
use std::f32::consts::PI;

use glam::IVec2;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use luisa::lang::types::vector::*;
use luisa::prelude::*;
use luisa_compute as luisa;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::light::{precomputed_lights, precomputed_slope_gathers};

mod light;

const GRID_SIZE: u32 = 64;
const SCALING: u32 = 32;
const SCALE_POWER: u32 = 5;

const DIRECTIONS: u32 = 9;
const TOTAL_DIRECTIONS: u32 = DIRECTIONS * 4;

const LIGHT_STEP: f32 = 0.1;
const COLOR_STEP: f32 = 0.1;

const WALL_ABSORB: u32 = 0b01;
// const WALL_REFLECT: u32 = 0b10;
// TODO: Could actually use this for fog by allowing partial blurs.
const WALL_BLUR: u32 = 0b100;
const WALL_DIFFUSE: u32 = 0b10000;

const EPSILON: f32 = 0.001;

// const SOLID_WALLS: u32 = WALL_DIFFUSE; // & WALL_REFLECT

#[derive(Debug, Copy, Clone)]
enum Writer {
    Empty,
    Absorb,
    Blur,
    Diffuse,
}
impl Writer {
    fn wall(&self) -> u32 {
        match self {
            Self::Empty => 0,
            Self::Absorb => WALL_ABSORB,
            Self::Blur => WALL_BLUR,
            Self::Diffuse => WALL_DIFFUSE,
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum State {
    Normal,
    UpdatingLight,
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

    type Lights = Tex3d<Vec4<f32>>;

    let gather_data = precomputed_slope_gathers();
    let gathers = gather_data.gathers;
    let transmissions = gather_data.transmissions;
    let angles = gather_data.angles;
    assert!(angles.len() == DIRECTIONS as usize);

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
    let colors = device.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, GRID_SIZE, GRID_SIZE, 1);

    let draw_kernel = Kernel::<fn(Lights)>::new(
        &device,
        &track!(|lights| {
            let display_pos = dispatch_id().xy();
            let pos = display_pos >> SCALE_POWER;
            let color = Vec3::<f32>::var_zeroed();
            for i in 0..TOTAL_DIRECTIONS {
                *color += lights.read(pos.extend(i)).xyz();
            }

            // let wall_alpha = 0.0.var();
            // let wall_color = Vec3::var_zeroed();
            let w = walls.read(pos);
            if w == WALL_ABSORB {
                *color = Vec3::new(0.2, 0.2, 0.2);
            } else if w == WALL_DIFFUSE {
                *color = Vec3::splat(0.1) + color;
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
                if dispatch_id().z == 0 {
                    walls.write(pos, 0);
                }
            }),
        );

    let update_wall_kernel = Kernel::<fn(Vec2<u32>, u32, Vec3<f32>)>::new(
        &device,
        &track!(|pos, wall, color| {
            let pos = pos + dispatch_id().xy();
            walls.write(pos, wall);
            colors.write(pos, color.extend(1.0));
            for i in 0..TOTAL_DIRECTIONS {
                emission.write(pos.extend(i), Vec4::splat(0.0));
            }
        }),
    );

    fn rotate(face: u32, dir: IVec2) -> IVec2 {
        match face {
            // +x
            0 => dir,
            // -y
            1 => IVec2::new(dir.y, -dir.x),
            // -x
            2 => -dir,
            // y
            3 => IVec2::new(-dir.y, dir.x),
            _ => panic!("Invalid"),
        }
    }

    let update_kernel = Kernel::<fn(Lights, Lights)>::new(
        &device,
        &track!(|lights, next_lights| {
            let pos = dispatch_id().xy() + 1;

            let light = [Vec4::<f32>::splat(0.0); TOTAL_DIRECTIONS as usize].var();

            let w = walls.read(pos);
            if w == WALL_ABSORB {
                for i in 0..TOTAL_DIRECTIONS {
                    next_lights.write(pos.extend(i), Vec4::splat(0.0));
                }
                return;
            }

            let apply = |face: u32| {
                escape!({
                    for (i, gather) in gathers.iter().enumerate() {
                        track!({
                            let dir = gather.direction + face * DIRECTIONS;
                            let opos = (pos.cast_i32()
                                + Vec2::<i32>::from(rotate(face, gather.offset)))
                            .cast_u32();
                            let ow = walls.read(opos);
                            if w == WALL_DIFFUSE || ow != WALL_DIFFUSE {
                                let transmission = transmissions[i] * lights.read(opos.extend(dir));
                                *light[dir] += transmission;
                            }
                        })
                    }
                })
            };

            apply(0);
            apply(1);
            apply(2);
            apply(3);

            let delta_light = [Vec4::<f32>::splat(0.0); TOTAL_DIRECTIONS as usize].var();

            let try_wall = |face: u32| {
                let offset = rotate(face, IVec2::X);
                let wpos = (pos.cast_i32() + Vec2::<i32>::from(offset)).cast_u32();
                // TODO: Add reflect.
                if walls.read(wpos) == WALL_DIFFUSE {
                    let wall_color = colors.read(wpos);
                    let gathered_light = Vec4::<f32>::var_zeroed();
                    escape!({
                        for wall_face in 0..4 {
                            for (i, gather) in gathers.iter().enumerate() {
                                if rotate(wall_face, gather.offset) + offset == IVec2::ZERO {
                                    let dir = gather.direction + wall_face * DIRECTIONS;
                                    track!({
                                        let lost_light = transmissions[i] * light[dir];
                                        *gathered_light += lost_light;
                                    });
                                }
                            }
                        }
                    });
                    *gathered_light *= wall_color;
                    escape!({
                        let mut normalization_factor = 0.0;
                        let face_angle = PI + (PI / 2.0 * face as f32);
                        for wall_face in 0..4 {
                            for i in 0..DIRECTIONS {
                                let angle = angles[i as usize] + wall_face as f32 * PI / 2.0;
                                if (angle - face_angle).cos() > EPSILON {
                                    normalization_factor += (angle - face_angle).cos()
                                        * precomputed_lights()[i as usize];
                                }
                            }
                        }
                        for wall_face in 0..4 {
                            for i in 0..DIRECTIONS {
                                let angle = angles[i as usize] + wall_face as f32 * PI / 2.0;
                                if (angle - face_angle).cos() > EPSILON {
                                    track!({
                                        *delta_light[i + wall_face * DIRECTIONS] +=
                                            (angle - face_angle).cos() * gathered_light
                                                / normalization_factor
                                                * precomputed_lights()[i as usize];
                                    });
                                }
                            }
                        }
                    });
                }
            };
            if w != WALL_DIFFUSE && w != WALL_ABSORB {
                try_wall(0);
                try_wall(1);
                try_wall(2);
                try_wall(3);
            }
            if w == WALL_BLUR {
                let total_light = Vec4::<f32>::var_zeroed();
                for i in 0..TOTAL_DIRECTIONS {
                    *total_light += light[i];
                }
                *total_light /= TOTAL_DIRECTIONS as f32;
                for i in 0..TOTAL_DIRECTIONS {
                    *light[i] = total_light;
                }
            }

            let color = colors.read(pos);

            for i in 0..TOTAL_DIRECTIONS {
                next_lights.write(pos.extend(i), (light[i] + delta_light[i]) * color);
            }
        }),
    );

    let mut parity = false;

    let mut cursor_pos = PhysicalPosition::new(0.0, 0.0);

    let mut active_buttons = HashSet::new();

    let mut light_color = Vec3::new(1.0, 0.7, 0.2);
    let mut emission = 0.3;

    let mut update_cursor = |active_buttons: &HashSet<MouseButton>,
                             cursor_pos: PhysicalPosition<f64>,
                             light_color: Vec3<f32>,
                             emission: f32,
                             writer: Writer| {
        let pos = Vec2::new(
            (cursor_pos.x as u32) >> SCALE_POWER,
            (cursor_pos.y as u32) >> SCALE_POWER,
        );
        if active_buttons.contains(&MouseButton::Right) {
            let total_light = [Vec3::new(
                light_color.x * emission,
                light_color.y * emission,
                light_color.z * emission,
            ); TOTAL_DIRECTIONS as usize];
            update_emission_kernel.dispatch([1, 1, TOTAL_DIRECTIONS], &pos, &total_light);
        } else if active_buttons.contains(&MouseButton::Left) {
            let w = writer.wall();

            update_wall_kernel.dispatch(
                [1, 1, 1],
                &pos,
                &writer.wall(),
                &if w == 0 {
                    Vec3::splat(1.0)
                } else {
                    Vec3::splat(0.9)
                },
            );
        }
    };
    let update_cursor = &mut update_cursor;

    let mut state = State::Normal;
    let mut writer = Writer::Absorb;

    /*
    E: Empty
    A: Absorb,
    B: Blur,
    D: Diffuse,
    */

    let mut update_keyboard =
        |ev: KeyEvent, light_color: &mut Vec3<f32>, emission: &mut f32, writer: &mut Writer| {
            if ev.state != ElementState::Pressed {
                return;
            }
            let PhysicalKey::Code(key) = ev.physical_key else {
                panic!("Invalid")
            };

            match state {
                State::Normal => match key {
                    KeyCode::KeyL => state = State::UpdatingLight,
                    KeyCode::KeyE => *writer = Writer::Empty,
                    KeyCode::KeyA => *writer = Writer::Absorb,
                    KeyCode::KeyB => *writer = Writer::Blur,
                    KeyCode::KeyD => *writer = Writer::Diffuse,
                    _ => (),
                },
                State::UpdatingLight => match key {
                    KeyCode::ArrowUp => {
                        *emission += LIGHT_STEP;
                    }
                    KeyCode::ArrowDown => {
                        *emission -= LIGHT_STEP;
                    }
                    KeyCode::Escape => state = State::Normal,
                    _ => (),
                },
            }
            match state {
                State::Normal => {
                    println!("{:?}", writer);
                }
                State::UpdatingLight => {
                    println!("Emission: {:?}, Color: {:?}", *emission, *light_color);
                }
            }
        };
    let update_keyboard = &mut update_keyboard;

    update_wall_kernel.dispatch(
        [GRID_SIZE, GRID_SIZE, 1],
        &Vec2::splat(0),
        &0,
        &Vec3::splat(1.0),
    );

    println!("{:?}", writer);

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
                    update_cursor(&active_buttons, cursor_pos, light_color, emission, writer);
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
                    update_cursor(&active_buttons, cursor_pos, light_color, emission, writer);
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    update_keyboard(event, &mut light_color, &mut emission, &mut writer);
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
