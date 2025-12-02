use ndarray_rand::rand::prelude::*;
use rust_ml::nn::sigmoidf;

#[rustfmt::skip]
const TRAIN: [(f32, f32);5] = [
    (0., 0.),
    (1., 2.),
    (2., 4.),
    (3., 6.),
    (4., 8.),
    ];
const SEED: u64 = 69;
const EPS: f32 = 1e-3;

fn cost(train: &[(f32, f32)], w: f32, b: f32) -> f32 {
    train
        .iter()
        .map(|(x, y)| (x * w + b, y))
        .fold(0., |acc, (x, y)| acc + (x - y).powi(2))
        / train.len() as f32
}

fn main() {
    let mut rng: StdRng = StdRng::seed_from_u64(SEED);
    let rate = 1e-3;
    let mut w: f32 = rng.random_range(0.0..10.0);
    let mut b: f32 = rng.random_range(0.0..5.0);
    println!("w: {w}");
    println!("b: {b}");
    TRAIN.iter().for_each(|(x, y)| {
        let x = x * w;
        println!("actual: {x}, expected: {y}")
    });

    println!("loss: {}", cost(&TRAIN, w, b));
    (0..10).for_each(|_| {
        let c = cost(&TRAIN, w, b);
        let dw = (cost(&TRAIN, w + EPS, b) - c) / EPS;
        let db = (cost(&TRAIN, w, b + EPS) - c) / EPS;
        w -= dw * rate;
        b -= db * rate;
        println!("loss: {}", cost(&TRAIN, w, b));
    });
    println!("w: {w}");
    println!("b: {b}");
}
