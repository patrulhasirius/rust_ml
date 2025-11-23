use bevy::prelude::*;
use rand::prelude::*;
use derive_more::{Sub, Mul};

#[rustfmt::skip]
const TRAIN: [(f32, f32, f32);4] = [
    (0., 0., 0.),
    (1., 0., 1.),
    (0., 1., 1.),
    (1., 1., 0.),
    ];
const SEED: u64 = 69;
const EPS: f32 = 1e-3;
const RATE: f32 = 1e-2;

#[derive(Debug, Clone, Copy, Sub, Mul)]
struct Xor {
    or_w1: f32,
    or_w2: f32,
    or_b: f32,

    nand_w1: f32,
    nand_w2: f32,
    nand_b: f32,

    and_w1: f32,
    and_w2: f32,
    and_b: f32,
}
impl Xor {
    fn new(mut rng: StdRng) -> Self {
        Self {
            or_w1: rng.random(),
            or_w2: rng.random(),
            or_b: rng.random(),

            nand_w1: rng.random(),
            nand_w2: rng.random(),
            nand_b: rng.random(),

            and_w1: rng.random(),
            and_w2: rng.random(),
            and_b: rng.random(),
        }
    }
    fn foward(self, x1: f32, x2: f32) -> f32 {
        let s = self;
        let a = sigmoidf(s.or_w1 * x1 + s.or_w2 * x2 + s.or_b);
        let b = sigmoidf(s.nand_w1 * x1 + s.nand_w2 * x2 + s.nand_b);
        sigmoidf(a * s.and_w1 + b * s.and_w2 + s.and_b)
    }
    fn cost(self) -> f32 {
        TRAIN
            .iter()
            //.map(|(x1, x2, y)| (x1 * w1, x2 * w2, y))
            .fold(0., |acc, (x1, x2, y)| {
                acc + (self.foward(*x1, *x2) - y).powi(2)
            })
            / TRAIN.len() as f32
    }
    fn finite_diff(self) -> Self {
        let c = self.cost();
        let mut m = self.clone();
        let mut g = m.clone();
        let mut saved: f32;

        saved = m.or_w1;
        m.or_w1 += EPS;
        g.or_w1 = (m.cost() - c) / EPS;
        m.or_w1 = saved;

        saved = m.or_w2;
        m.or_w2 += EPS;
        g.or_w2 = (m.cost() - c) / EPS;
        m.or_w2 = saved;

        saved = m.or_b;
        m.or_b += EPS;
        g.or_b = (m.cost() - c) / EPS;
        m.or_b = saved;

        saved = m.nand_w1;
        m.nand_w1 += EPS;
        g.nand_w1 = (m.cost() - c) / EPS;
        m.nand_w1 = saved;

        saved = m.nand_w2;
        m.nand_w2 += EPS;
        g.nand_w2 = (m.cost() - c) / EPS;
        m.nand_w2 = saved;

        saved = m.nand_b;
        m.nand_b += EPS;
        g.nand_b = (m.cost() - c) / EPS;
        m.nand_b = saved;

        saved = m.and_w1;
        m.and_w1 += EPS;
        g.and_w1 = (m.cost() - c) / EPS;
        m.and_w1 = saved;

        saved = m.and_w2;
        m.and_w2 += EPS;
        g.and_w2 = (m.cost() - c) / EPS;
        m.and_w2 = saved;

        saved = m.and_b;
        m.and_b += EPS;
        g.and_b = (m.cost() - c) / EPS;
        //m.and_b = saved;

        g
    }
    fn learn(self) -> Self {
        self - self.finite_diff() * RATE
    }
}

fn sigmoidf(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

fn main() {
    //let mut rng: StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut rng: StdRng = rand::rngs::StdRng::from_os_rng();
    let mut m = Xor::new(rng);
    dbg!(m);
    println!("cost: {}", m.cost());

    //TRAIN.iter().for_each(|(x1, x2, y)| {
    //    let x1 = x1 * w1;
    //    let x2 = x2 * w2;
    //    println!("x1: {x1}, x2: {x2}, y: {y}");
    //});

    (0..(1000*100)).for_each(|_| {
        m = m.learn();
    });

    dbg!(m);
    println!("cost: {}", m.cost());

    TRAIN.iter().for_each(|(x1, x2, _)| {
        let y = m.foward(*x1, *x2);
        println!("x1: {x1}, x2: {x2}, y: {y}");
    });
}
