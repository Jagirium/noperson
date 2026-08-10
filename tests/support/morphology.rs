#[derive(Debug, Clone, Copy)]
pub struct MorphologyFixture {
    pub name: &'static str,
    pub width: usize,
    pub height: usize,
    pub input: &'static [f32],
}

pub const AMOUNTS: [i32; 7] = [-100, -3, -1, 0, 1, 2, 100];

const ONE_BY_ONE: [f32; 1] = [0.25];
const ONE_BY_N: [f32; 4] = [0.0, 1.0, 0.0, 0.0];
const TWO_BY_TWO: [f32; 4] = [0.0, 1.0, 0.5, 0.25];
const EDGE_IMPULSE: [f32; 9] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
const CENTER_IMPULSE: [f32; 25] = [
    0.0, 0.0, 0.0, 0.0, 0.0, // row 0
    0.0, 0.0, 0.0, 0.0, 0.0, // row 1
    0.0, 0.0, 1.0, 0.0, 0.0, // row 2
    0.0, 0.0, 0.0, 0.0, 0.0, // row 3
    0.0, 0.0, 0.0, 0.0, 0.0, // row 4
];
const MONOTONIC_GRADIENT: [f32; 12] = [
    0.0, 0.1, 0.2, 0.3, // row 0
    0.4, 0.5, 0.6, 0.7, // row 1
    0.8, 0.9, 1.0, 1.0, // row 2
];

pub const FIXTURES: [MorphologyFixture; 6] = [
    MorphologyFixture {
        name: "1x1",
        width: 1,
        height: 1,
        input: &ONE_BY_ONE,
    },
    MorphologyFixture {
        name: "1xN",
        width: 4,
        height: 1,
        input: &ONE_BY_N,
    },
    MorphologyFixture {
        name: "2x2",
        width: 2,
        height: 2,
        input: &TWO_BY_TWO,
    },
    MorphologyFixture {
        name: "edge impulse",
        width: 3,
        height: 3,
        input: &EDGE_IMPULSE,
    },
    MorphologyFixture {
        name: "center impulse",
        width: 5,
        height: 5,
        input: &CENTER_IMPULSE,
    },
    MorphologyFixture {
        name: "monotonic gradient",
        width: 4,
        height: 3,
        input: &MONOTONIC_GRADIENT,
    },
];

pub fn scalar_reference(input: &[f32], width: usize, height: usize, amount: i32) -> Vec<f32> {
    assert_eq!(input.len(), width * height);
    let radius = amount.unsigned_abs().min(100) as usize;
    let mut output = vec![0.0; input.len()];
    for y in 0..height {
        for x in 0..width {
            let mut maximum = 0.0f32;
            for offset_y in -(radius as isize)..=(radius as isize) {
                let source_y = (y as isize + offset_y).clamp(0, height as isize - 1) as usize;
                for offset_x in -(radius as isize)..=(radius as isize) {
                    let source_x = (x as isize + offset_x).clamp(0, width as isize - 1) as usize;
                    let value = input[source_y * width + source_x];
                    maximum = maximum.max(if amount < 0 { 1.0 - value } else { value });
                }
            }
            output[y * width + x] = if amount < 0 { 1.0 - maximum } else { maximum };
        }
    }
    output
}

pub fn repeated_clamped_max(input: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut current = input.to_vec();
    for _ in 0..radius {
        let mut next = vec![0.0; input.len()];
        for y in 0..height {
            for x in 0..width {
                let mut maximum = 0.0f32;
                for offset_y in -1isize..=1 {
                    let source_y = (y as isize + offset_y).clamp(0, height as isize - 1) as usize;
                    for offset_x in -1isize..=1 {
                        let source_x =
                            (x as isize + offset_x).clamp(0, width as isize - 1) as usize;
                        maximum = maximum.max(current[source_y * width + source_x]);
                    }
                }
                next[y * width + x] = maximum;
            }
        }
        current = next;
    }
    current
}
