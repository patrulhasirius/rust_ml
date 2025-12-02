use ndarray::{Array, array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use rust_ml::nn::sigmoidf;

#[rustfmt::skip]
const TRAIN: [(f32, f32, f32);4] = [
    (0., 0., 0.),
    (1., 0., 1.),
    (0., 1., 1.),
    (1., 1., 0.),
    ];
const SEED: u64 = 69;
const EPS: f32 = 1e-3;
const RATE: f32 = 1e-1;

fn main() {
    //let mut rng: StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut rng: StdRng = StdRng::from_os_rng();
    let x = array![[0., 1.]];
    let w1 = Array::random_using((2, 2), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);
    let b1 = Array::random_using((1, 2), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);

    let w2 = Array::random_using((2, 1), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);
    let b2 = Array::random_using((1, 1), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);

    dbg!(&x);
    dbg!(&w1);
    dbg!(&b1);
    dbg!(&w2);
    dbg!(&b2);

    let mut a1 = x.dot(&w1);
    a1 = a1 + b1;
    a1 = a1.mapv_into(sigmoidf);

    let mut a2 = a1.dot(&w2);
    a2 = a2 + b2;
    a2 = a2.mapv_into(sigmoidf);

    dbg!(a2.get((0,0)).unwrap());

    //println!("cost: {}", m.cost());

    ////TRAIN.iter().for_each(|(x1, x2, y)| {
    ////    let x1 = x1 * w1;
    ////    let x2 = x2 * w2;
    ////    println!("x1: {x1}, x2: {x2}, y: {y}");
    ////});

    //(0..(1000 * 100)).for_each(|_| {
    //    m = m.learn();
    //});

    //dbg!(m);
    //println!("cost: {}", m.cost());

    //TRAIN.iter().for_each(|(x1, x2, _)| {
    //    let y = m.foward(*x1, *x2);
    //    println!("x1: {x1}, x2: {x2}, y: {y}");
    //});
}
