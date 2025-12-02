use ndarray_rand::rand::prelude::*;
use rust_ml::nn::sigmoidf;

#[rustfmt::skip]
const TRAIN: [(f32, f32, f32);4] = [
    (0., 0., 1.),
    (1., 0., 1.),
    (0., 1., 1.),
    (1., 1., 0.),
    ];
const SEED: u64 = 69;
const EPS: f32 = 1e-3;

fn cost(train: &[(f32, f32, f32)], w1: f32, w2: f32, b: f32) -> f32 {
    train
        .iter()
        .map(|(x1, x2, y)| (x1 * w1, x2 * w2, y))
        .fold(0., |acc, (x1, x2, y)| {
            acc + (sigmoidf(x1 + x2 + b) - y).powi(2)
        })
        / train.len() as f32
}

fn main() {
    //let mut rng: StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut rng: StdRng = StdRng::from_os_rng();
    let rate = 1e-1;
    let mut w1: f32 = rng.random_range(-5.0..5.0);
    let mut w2: f32 = rng.random_range(-5.0..5.0);
    let mut b: f32 = rng.random_range(-5.0..5.0);
    println!("w1: {w1}");
    println!("w2: {w2}");
    println!("w2: {b}");
    TRAIN.iter().for_each(|(x1, x2, y)| {
        let x1 = x1 * w1;
        let x2 = x2 * w2;
        println!("x1: {x1}, x2: {x2}, y: {y}");
    });

    println!("loss: {}", cost(&TRAIN, w1, w2, b));
    (0..(1000 * 1000)).for_each(|_| {
        let c = cost(&TRAIN, w1, w2, b);
        let dw1 = (cost(&TRAIN, w1 + EPS, w2, b) - c) / EPS;
        let dw2 = (cost(&TRAIN, w1, w2 + EPS, b) - c) / EPS;
        let db = (cost(&TRAIN, w1, w2, b + EPS) - c) / EPS;
        w1 -= dw1 * rate;
        w2 -= dw2 * rate;
        b -= db * rate;
        //println!("loss: {}", cost(&TRAIN, w1, w2, b));
    });
    println!("w1: {w1}");
    println!("w2: {w2}");
    println!("b: {b}");
    println!("loss: {}", cost(&TRAIN, w1, w2, b));

    TRAIN.iter().for_each(|(x1, x2, _)| {
        println!("{} | {} = {}", x1, x2, sigmoidf(x1 * w1 + x2 * w2 + b));
    });
}
