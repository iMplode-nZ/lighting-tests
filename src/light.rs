use glam::IVec2;

use super::*;

// Face of direction is +x by default.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Gather {
    pub offset: IVec2,
    pub direction: u32,
}

#[derive(Debug, Clone)]
pub struct GatherData {
    pub gathers: Vec<Gather>,
    pub angles: Vec<f32>,
    pub blurs: Vec<f32>,
    pub transmissions: Vec<f32>,
}

pub fn compute_index_transmissions(angle: f32, blur: f32) -> [f32; 3] {
    let slope = angle.tan();
    let y = slope / (1.0 + slope.abs());
    let x = 1.0 / (1.0 + slope.abs());

    let sg = y.is_sign_positive();
    let n = x * (1.0 - 3.0 * blur) + blur;
    let tp = y.abs() * (1.0 - 3.0 * blur) + blur;
    let tm = blur;
    if sg {
        [n, tp, tm]
    } else {
        [n, tm, tp]
    }
}

pub fn compute_slope_gathers_n(directions: u32) -> GatherData {
    let mut gathers = vec![];
    let mut angles = vec![];
    let mut blurs = vec![];
    let mut transmissions = vec![];

    for dir in 0..directions {
        let angle = ((dir as f32 + 0.5) / directions as f32 - 0.5) * PI / 2.0;
        println!("Angle: {:?}", angle);

        let blur = if dir == 4 {
            0.1
        } else if dir == 3 || dir == 5 {
            0.04
        } else {
            0.0
        };

        angles.push(angle);
        blurs.push(blur);

        let tr = compute_index_transmissions(angle, blur);
        transmissions.extend_from_slice(&tr);

        gathers.push(Gather {
            offset: IVec2::new(-1, 0),
            direction: dir,
        });
        gathers.push(Gather {
            offset: IVec2::new(0, 1),
            direction: dir,
        });
        gathers.push(Gather {
            offset: IVec2::new(0, -1),
            direction: dir,
        });
    }

    GatherData {
        gathers,
        angles,
        blurs,
        transmissions,
    }
}
/*
Angles: [-0.69813174, -0.52359873, -0.3890658, -0.21453294, 0.0, 0.21453294, 0.3890658, 0.52359873, 0.6981317]
Lights: [4, 4, 4, 3.8, 5, 3.8, 4, 4, 4]
Blurs: [0.0, 0.0, 0.07, 0.12, 0.2, 0.12, 0.07, 0.0, 0.0]
*/
pub fn precomputed_lights() -> [f32; 9] {
    [4.0, 4.0, 4.0, 3.8, 5.0, 3.8, 4.0, 4.0, 4.0].map(|x| x / 4.0)
}
pub fn precomputed_slope_gathers(directions: u32) -> GatherData {
    assert!(directions == 9);
    let mut gathers = vec![];
    let angles = vec![
        -0.69813174,
        -0.52359873,
        -0.3890658,
        -0.21453294,
        0.0,
        0.21453294,
        0.3890658,
        0.52359873,
        0.6981317,
    ];
    let blurs = vec![0.0, 0.0, 0.07, 0.12, 0.2, 0.12, 0.07, 0.0, 0.0];
    let mut transmissions = vec![];

    for dir in 0..directions {
        let tr = compute_index_transmissions(angles[dir as usize], blurs[dir as usize]);
        transmissions.extend_from_slice(&tr);

        gathers.push(Gather {
            offset: IVec2::new(-1, 0),
            direction: dir,
        });
        gathers.push(Gather {
            offset: IVec2::new(0, 1),
            direction: dir,
        });
        gathers.push(Gather {
            offset: IVec2::new(0, -1),
            direction: dir,
        });
    }

    GatherData {
        gathers,
        angles,
        blurs,
        transmissions,
    }
}
