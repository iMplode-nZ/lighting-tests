use glam::IVec2;

use super::*;

// Face of direction is +x by default.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Gather {
    pub offset: IVec2,
    pub direction: u32,
    pub quantity: f32,
}

pub fn compute_simple_gathers() -> (Vec<Gather>, u32) {
    let center_fraction = 1.0_f32.atan2(3.0) / (PI / 4.0);
    let side_fraction = (1.0 - center_fraction) / 2.0;

    (
        vec![
            Gather {
                offset: IVec2::new(-1, 0),
                direction: 0,
                quantity: center_fraction,
            },
            Gather {
                offset: IVec2::new(-1, -1),
                direction: 0,
                quantity: side_fraction,
            },
            Gather {
                offset: IVec2::new(-1, 1),
                direction: 0,
                quantity: side_fraction,
            },
        ],
        1,
    )
}

pub fn compute_gathers_2() -> (Vec<Gather>, u32) {
    (
        vec![
            Gather {
                offset: IVec2::new(-1, 0),
                direction: 0,
                quantity: 0.5,
            },
            Gather {
                offset: IVec2::new(0, -1),
                direction: 0,
                quantity: 0.5,
            },
            Gather {
                offset: IVec2::new(-1, 0),
                direction: 1,
                quantity: 0.4,
            },
            Gather {
                offset: IVec2::new(-1, -1),
                direction: 1,
                quantity: 0.3,
            },
            Gather {
                offset: IVec2::new(-1, 1),
                direction: 1,
                quantity: 0.3,
            },
        ],
        2,
    )
}

pub fn compute_slope_gathers_n(directions: u32) -> Vec<Gather> {
    let mut gathers = vec![];

    for dir in 0..directions {
        let angle = ((dir as f32 + 0.5) / directions as f32 - 0.5) * PI / 2.0;
        println!("Angle: {:?}", angle);

        let blur = 0.0; // if dir == directions / 2 { 0.05 } else { 0.0 }; // 0.1 * (1.0 - (angle / (PI / 4.0)) * (angle / (PI / 4.0)));

        let slope = angle.tan();
        let y = slope / (1.0 + slope.abs());
        let x = 1.0 / (1.0 + slope.abs());
        gathers.push(Gather {
            offset: IVec2::new(-1, 0),
            direction: dir,
            quantity: x * (1.0 - 3.0 * blur) + blur,
        });
        gathers.push(Gather {
            offset: IVec2::new(0, y.signum() as i32),
            direction: dir,
            quantity: y.abs() * (1.0 - 3.0 * blur) + blur,
        });
        gathers.push(Gather {
            offset: IVec2::new(0, -y.signum() as i32),
            direction: dir,
            quantity: blur,
        });
    }

    gathers
}
