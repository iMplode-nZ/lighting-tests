use std::collections::HashSet;
use std::env::current_exe;
use std::f32::consts::PI;
use std::io::Write;
use std::{array, io};

use glam::IVec2;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use luisa::lang::types::vector::*;
use luisa::prelude::*;
use luisa_compute as luisa;
use winit::keyboard::{KeyCode, PhysicalKey};

use crate::light::{
    compute_index_transmissions, compute_slope_gathers_n, precomputed_lights,
    precomputed_slope_gathers,
};

mod light;

const GRID_SIZE: u32 = 64;
const SCALING: u32 = 32;
const SCALE_POWER: u32 = 5;

const DIRECTIONS: u32 = 9;
const TOTAL_DIRECTIONS: u32 = DIRECTIONS * 4;

const BLUR_STEP: f32 = 0.01;
const ANGLE_STEP: f32 = 0.01;
const LIGHT_STEP: f32 = 0.1;
const TRANSMISSION_STEP: f32 = 0.01;

const WALL_ABSORB: u32 = 0b01;
// const WALL_REFLECT: u32 = 0b10;
// TODO: Could actually use this for fog by allowing partial blurs.
const WALL_BLUR: u32 = 0b100;
const WALL_TINT: u32 = 0b1000;
const WALL_DIFFUSE: u32 = 0b10000;

const EPSILON: f32 = 0.001;

// const SOLID_WALLS: u32 = WALL_DIFFUSE; // & WALL_REFLECT

#[derive(Debug, Copy, Clone)]
enum State {
    Normal,
    UpdatingBlur,
    UpdatingAngle,
    UpdatingTransmission(u32),
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

    let normal_light_color: Vec3<f32> = Vec3::new(1.0, 0.7, 0.2);
    let active_light_color: Vec3<f32> = Vec3::new(0.3, 0.8, 0.9);

    let gather_data = precomputed_slope_gathers(DIRECTIONS);
    let gathers = gather_data.gathers;
    let mut transmissions = gather_data.transmissions;
    let mut angles = gather_data.angles;
    let mut blurs = gather_data.blurs;

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
    let wall_colors =
        device.create_tex2d::<Vec4<f32>>(PixelStorage::Float4, GRID_SIZE, GRID_SIZE, 1);

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
            } else if w == WALL_BLUR {
                *color = Vec3::new(0.7, 0.7, 0.7);
            } else if w == WALL_TINT {
                *color = wall_colors.read(pos).xyz();
            } else if w == WALL_DIFFUSE {
                *color = Vec3::new(0.4, 0.4, 0.4);
            }
            // if w == WALL_ABSORB {
            //     *wall_color = Vec3::new(0.2, 0.2, 0.2);
            //     *wall_
            // } else if w == WALL_BLUR {
            //     *wall_color = Vec3::new(0.7, 0.7, 0.7);
            //     *wall_alpha = 0.3;
            // } else if w == WALL_TINT {
            //     *wall_color = wall_colors.read(pos);
            //     *wall_alpha = 0.1;
            // } else if w == WALL_DIFFUSE {
            //     *color = Vec3::new(0.0, 0.0, 1.0).expr();
            // }

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

    let update_wall_kernel = Kernel::<fn(Vec2<u32>, u32, Vec3<f32>)>::new(
        &device,
        &track!(|pos, wall, color| {
            walls.write(pos, wall);
            wall_colors.write(pos, color.extend(1.0));
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

    let update_kernel = Kernel::<fn(Lights, Lights, [f32; 3 * DIRECTIONS as usize])>::new(
        &device,
        &track!(|lights, next_lights, transmissions2| {
            let pos = dispatch_id().xy() + 1;

            let light = [Vec4::<f32>::splat(0.0); TOTAL_DIRECTIONS as usize].var();

            let w = walls.read(pos);
            if w == WALL_ABSORB || w == WALL_DIFFUSE {
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
                            let transmission = transmissions2[i as u32]
                                * lights.read(
                                    (pos.cast_i32()
                                        + Vec2::<i32>::from(rotate(face, gather.offset)))
                                    .cast_u32()
                                    .extend(dir),
                                );
                            *light[dir] += transmission;
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
                    let wall_color = wall_colors.read(wpos);
                    let gathered_light = Vec4::<f32>::var_zeroed();
                    escape!({
                        for wall_face in 0..4 {
                            for (i, gather) in gathers.iter().enumerate() {
                                // TODO: This stuff seems buggy. Fix.
                                if rotate(wall_face, gather.offset) + offset == IVec2::ZERO {
                                    let dir = gather.direction + wall_face * DIRECTIONS;
                                    println!(
                                        "Adding: Direction {:?}, Face: {:?}, Gather Offset: {:?}, Transmission: {:?}",
                                        gather.direction, wall_face, gather.offset, transmissions[i],
                                    );
                                    track!({
                                        let lost_light = transmissions2[i as u32] * light[dir];
                                        // IS THIS NECESSARY?
                                        // *delta_light[dir] -= lost_light;
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
                                    normalization_factor += (angle - face_angle).cos();
                                }
                            }
                        }
                        for wall_face in 0..4 {
                            for i in 0..DIRECTIONS {
                                let angle = angles[i as usize] + wall_face as f32 * PI / 2.0;
                                if (angle - face_angle).cos() > EPSILON {
                                    track!({
                                        *delta_light[i + wall_face * DIRECTIONS] +=
                                            (angle - face_angle.cos()) / normalization_factor
                                                * gathered_light;
                                    });
                                }
                            }
                        }
                    });
                }
            };
            try_wall(0);
            try_wall(1);
            try_wall(2);
            try_wall(3);

            // TODO: Adjust for non-equal light directions.
            // if w == WALL_BLUR {
            //     let total_light = Vec4::<f32>::var_zeroed();
            //     for i in 0..TOTAL_DIRECTIONS {
            //         *total_light += light[i];
            //     }
            //     *total_light /= TOTAL_DIRECTIONS as f32;
            //     for i in 0..TOTAL_DIRECTIONS {
            //         *light[i] = total_light;
            //     }
            // }

            for i in 0..TOTAL_DIRECTIONS {
                next_lights.write(pos.extend(i), light[i] + delta_light[i]);
            }
        }),
    );

    let mut parity = false;

    let mut cursor_pos = PhysicalPosition::new(0.0, 0.0);

    let mut active_buttons = HashSet::new();

    let pl = precomputed_lights();
    let mut light: [f32; TOTAL_DIRECTIONS as usize] =
        array::from_fn(|i| 0.3 * pl[i % DIRECTIONS as usize]);

    let mut light_pos = Vec2::splat(0_u32);
    let mut activate_light = false;
    let mut activated_light: u32 = 0;

    let update_light =
        |light: [f32; TOTAL_DIRECTIONS as usize], light_pos, activate_light, activated_light| {
            let mut total_light = [Vec3::splat(0.0); TOTAL_DIRECTIONS as usize];
            for (i, lm) in light.iter().enumerate() {
                let color = if activated_light == i as u32 && activate_light {
                    active_light_color
                } else {
                    normal_light_color
                };
                total_light[i] = Vec3::new(color.x * lm, color.y * lm, color.z * lm);
            }
            update_emission_kernel.dispatch([1, 1, TOTAL_DIRECTIONS], &light_pos, &total_light);
        };

    let mut update_cursor = |active_buttons: &HashSet<MouseButton>,
                             cursor_pos: PhysicalPosition<f64>,
                             light_pos: &mut Vec2<u32>| {
        let pos = Vec2::new(
            (cursor_pos.x as u32) >> SCALE_POWER,
            (cursor_pos.y as u32) >> SCALE_POWER,
        );
        if active_buttons.contains(&MouseButton::Left) {
            // update_emission_kernel.dispatch(
            //     [1, 1, TOTAL_DIRECTIONS],
            //     light_pos,
            //     &[Vec3::splat(0.0); TOTAL_DIRECTIONS as usize],
            // );
            *light_pos = pos;
            true
        } else {
            if active_buttons.contains(&MouseButton::Right) {
                update_wall_kernel.dispatch([1, 1, 1], &pos, &WALL_ABSORB, &Vec3::splat(0.0));
            }
            if active_buttons.contains(&MouseButton::Middle) {
                update_wall_kernel.dispatch([1, 1, 1], &pos, &WALL_DIFFUSE, &Vec3::splat(1.0));
            }
            false
        }
    };
    let update_cursor = &mut update_cursor;

    let mut number_buffer = String::new();

    let mut state = State::Normal;

    /*
    [ESC]: Reset
    L: Toggle light activation (blue)
    Up/down to change direction by one.
    B: Enter blur change context (up/down to change).
    A: Enter angle change context (up/down to change).
    P: Print this direction's transmissions.
    T: Enter single transmission change context (specify 0/1/2 offset).
    E: Change emission (specify direction), or up/down to change direction by one.
    D: Print all the things.
    */

    let mut update_keyboard = |ev: KeyEvent,
                               transmissions: &mut Vec<f32>,
                               light: &mut [f32; TOTAL_DIRECTIONS as usize],
                               activate_light: &mut bool,
                               activated_light: &mut u32| {
        if ev.state != ElementState::Pressed {
            return;
        }
        if let Some(text) = ev.text {
            if text.is_ascii() {
                print!("{}", text);
                let _ = io::stdout().flush();
            }
            let text = text.as_str();
            if text.len() == 1 && "-0123456789.".contains(text) {
                number_buffer.push_str(text);
                return;
            }
        }
        let PhysicalKey::Code(key) = ev.physical_key else {
            panic!("Invalid")
        };

        println!();

        let dir = (*activated_light % DIRECTIONS) as usize;

        let mut num = None;

        match key {
            KeyCode::Escape => {
                state = State::Normal;
                number_buffer.clear();
            }
            KeyCode::KeyL => {
                *activate_light = !*activate_light;
            }
            KeyCode::KeyP => {
                println!(
                    "Angle: {:?}\nBlur: {:?}\nTransmissions: {:?}",
                    angles[dir],
                    blurs[dir],
                    &transmissions[dir * 3..dir * 3 + 3]
                );
            }
            KeyCode::KeyD => {
                println!(
                    "Angles: {:?}\nBlurs: {:?}\nTransmissions: {:?}",
                    angles, blurs, transmissions
                );
            }
            KeyCode::Enter => {
                num = Some(number_buffer.parse::<f32>().unwrap());
                number_buffer.clear();
            }
            _ => (),
        }
        let mut updated = false;
        match state {
            State::Normal => {
                if let Some(num) = num {
                    if num.fract() == 0.0 && num >= 0.0 && num < TOTAL_DIRECTIONS as f32 {
                        *activated_light = num as u32;
                    }
                }
                match key {
                    KeyCode::ArrowUp => {
                        *activated_light = (*activated_light + 1) % TOTAL_DIRECTIONS;
                    }
                    KeyCode::ArrowDown => {
                        *activated_light =
                            (*activated_light + TOTAL_DIRECTIONS - 1) % TOTAL_DIRECTIONS;
                    }
                    KeyCode::KeyA => {
                        state = State::UpdatingAngle;
                    }
                    KeyCode::KeyB => {
                        state = State::UpdatingBlur;
                    }
                    KeyCode::KeyT => {
                        state = State::UpdatingTransmission(0);
                    }
                    KeyCode::KeyE => {
                        state = State::UpdatingLight;
                    }
                    _ => (),
                }
            }
            State::UpdatingAngle => {
                if let Some(num) = num {
                    angles[dir] = num;
                    updated = true;
                }
                match key {
                    KeyCode::ArrowUp => {
                        angles[dir] += ANGLE_STEP;
                        updated = true;
                    }
                    KeyCode::ArrowDown => {
                        angles[dir] -= ANGLE_STEP;
                        updated = true;
                    }
                    _ => (),
                }
            }
            State::UpdatingBlur => {
                if let Some(num) = num {
                    blurs[dir] = num;
                    updated = true;
                }
                match key {
                    KeyCode::ArrowUp => {
                        blurs[dir] += BLUR_STEP;
                        updated = true;
                    }
                    KeyCode::ArrowDown => {
                        blurs[dir] -= BLUR_STEP;
                        updated = true;
                    }
                    _ => (),
                }
            }
            State::UpdatingTransmission(idx) => {
                let t = dir * 3 + idx as usize;
                if let Some(num) = num {
                    transmissions[t] = num;
                }
                match key {
                    KeyCode::ArrowUp => {
                        transmissions[t] += TRANSMISSION_STEP;
                    }
                    KeyCode::ArrowDown => {
                        transmissions[t] -= TRANSMISSION_STEP;
                    }
                    KeyCode::ArrowLeft => {
                        state = State::UpdatingTransmission((idx + 2) % 3);
                    }
                    KeyCode::ArrowRight => {
                        state = State::UpdatingTransmission((idx + 1) % 3);
                    }
                    _ => (),
                }
            }
            State::UpdatingLight => {
                if let Some(num) = num {
                    light[*activated_light as usize] = num;
                }
                match key {
                    KeyCode::ArrowUp => {
                        light[*activated_light as usize] += LIGHT_STEP;
                    }
                    KeyCode::ArrowDown => {
                        light[*activated_light as usize] -= LIGHT_STEP;
                    }
                    _ => (),
                }
            }
        }
        if updated {
            let tr = compute_index_transmissions(angles[dir], blurs[dir]);
            transmissions[dir * 3..dir * 3 + 3].copy_from_slice(&tr);
        }
        let dir = (*activated_light % DIRECTIONS) as usize;
        match state {
            State::Normal => {
                println!("Direction: {:?}", dir);
            }
            State::UpdatingAngle => {
                println!("Direction: {:?}, Angle: {:?}", dir, angles[dir]);
            }
            State::UpdatingBlur => {
                println!("Direction: {:?}, Blur: {:?}", dir, blurs[dir]);
            }
            State::UpdatingTransmission(idx) => {
                println!(
                    "Direction: {:?}, Index: {:?} (Offset: {:?})",
                    dir,
                    idx,
                    gathers[dir * 3 + idx as usize].offset
                );
            }
            State::UpdatingLight => {
                println!(
                    "Facing: {:?}, Emission: {:?}",
                    *activated_light, light[*activated_light as usize]
                );
            }
        }
        print!("> ");
        let _ = io::stdout().flush();
    };
    let update_keyboard = &mut update_keyboard;

    println!("Ready");
    println!("> ");

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
                                &array::from_fn(|i| transmissions[i]),
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
                    if update_cursor(&active_buttons, cursor_pos, &mut light_pos) {
                        update_light(light, light_pos, activate_light, activated_light);
                    }
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
                    if update_cursor(&active_buttons, cursor_pos, &mut light_pos) {
                        update_light(light, light_pos, activate_light, activated_light);
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    update_keyboard(
                        event,
                        &mut transmissions,
                        &mut light,
                        &mut activate_light,
                        &mut activated_light,
                    );
                    update_light(light, light_pos, activate_light, activated_light);
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
