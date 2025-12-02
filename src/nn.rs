use ndarray::Array;
pub fn sigmoidf(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}
