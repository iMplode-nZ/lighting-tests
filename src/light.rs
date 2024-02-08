use glam::Vec2;

use super::*;

const EPSILON: f32 = 0.001;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Gather {
    // Of the direction to.
    pub face_offset: i8,
    pub direction_to: u32,
    pub direction_from: u32,
    pub quantity: f32,
}

pub fn compute_gathers(directions: u32) -> Vec<Gather> {
    let gather_angle = |face_offset: i8, section: f32| {
        let angle = (section - 0.5) * PI / 2.0;
        let pos = match face_offset {
            0 => Vec2::new(1.0, angle.tan()),
            1 => Vec2::new(-angle.tan(), 1.0),
            -1 => Vec2::new(angle.tan(), -1.0),
            _ => panic!("Invalid face offset"),
        };
        let origin = Vec2::new(-2.0, 0.0);
        let delta = pos - origin;
        let out_angle = delta.y.atan2(delta.x);
        out_angle / (PI / 2.0) + 0.5
    };
    let gather_direction = |face_offset: i8, dir: f32| {
        gather_angle(face_offset, dir / directions as f32) * directions as f32
    };
    let mut gathers = vec![];
    for face_offset in -1_i8..=1 {
        for dir in 0..directions {
            let mut start = gather_direction(face_offset, dir as f32);
            let mut end = gather_direction(face_offset, (dir + 1) as f32);
            if start.ceil() - start < EPSILON {
                start = start.ceil() + EPSILON;
            }
            if end - end.floor() < EPSILON {
                end = end.floor() - EPSILON;
            }
            assert!(start < end);
            assert!(start >= 0.0);
            assert!(end < directions as f32);
            assert!(
                (start.floor() == end.floor() && start.ceil() == end.ceil())
                    || start.ceil() == end.floor()
            );
            // Split
            if start.ceil() == end.floor() {
                gathers.push(Gather {
                    face_offset,
                    direction_to: dir,
                    direction_from: start.floor() as u32,
                    quantity: start.ceil() - start,
                });
                gathers.push(Gather {
                    face_offset,
                    direction_to: dir,
                    direction_from: end.floor() as u32,
                    quantity: end - end.floor(),
                });
            } else {
                gathers.push(Gather {
                    face_offset,
                    direction_to: dir,
                    direction_from: start.floor() as u32,
                    quantity: end - start,
                });
            }
        }
    }
    gathers
}

// #[test]
// fn test_gathers() {
//     panic!("{:#?}", compute_gathers(2));
// }
