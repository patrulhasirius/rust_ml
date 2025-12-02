use ndarray_rand::rand::prelude::*;
use ndarray::{Array, Dim, Ix, IxDyn, array};
use rust_ml::nn::sigmoidf;
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;

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

type NormalArray = Array<f32, Dim<[usize; 2]>>;

#[derive(Debug, Clone)]
struct Xor {
    a0: NormalArray,
    w1: NormalArray,
    b1: NormalArray,
    w2: NormalArray,
    b2: NormalArray,

}
impl Xor {
    fn new(a0: NormalArray, w1: NormalArray, b1: NormalArray, w2: NormalArray, b2: NormalArray) -> Self {
        Xor{
            a0,
            w1,
            b1,
            w2,
            b2,
        }
    }
    fn foward(&self) -> f32 {
        let s = self.clone();
        let mut a1 = s.a0.dot(&s.w1);
        a1 = a1 + s.b1;
        a1 = a1.mapv_into(sigmoidf);

        let mut a2 = a1.dot(&s.w2);
        a2 = a2 + s.b2;
        a2 = a2.mapv_into(sigmoidf);
        *a2.get((0,0)).unwrap()
    }
    //fn learn(self) -> Self {
    //    self - self.finite_diff() * RATE
    //}
}

fn main() {
    //let mut rng: StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut rng: StdRng = StdRng::from_os_rng();
    let a0 = array![[0., 1.]];
    let w1 = Array::random_using((2, 2), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);
    let b1 = Array::random_using((1, 2), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);

    let w2 = Array::random_using((2, 1), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);
    let b2 = Array::random_using((1, 1), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);

    let mut m = Xor::new(a0, w1, b1, w2, b2);

    dbg!(&m);

    //TRAIN.iter().for_each(|(x1, x2, y)| {
    //    let x1 = x1 * w1;
    //    let x2 = x2 * w2;
    //    println!("x1: {x1}, x2: {x2}, y: {y}");
    //});

    //(0..(1000 * 100)).for_each(|_| {
    //    m = m.learn();
    //});

    //dbg!(m);
    //println!("cost: {}", m.cost());

    TRAIN.iter().for_each(|(x1, x2, _)| {
        m.a0 = array![[*x1, *x2]];
        let y = m.foward();
        println!("x1: {x1}, x2: {x2}, y: {y}");
    });
}
