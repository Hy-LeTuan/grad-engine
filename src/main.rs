pub mod config;
pub mod graph;
pub mod ops;
pub mod tensor_core;
pub mod utils;

use ndarray::Axis;
use tensor_core::tensor::Tensor;

// use graph::visualizer::Visualizer;
// use crate::graph::visualizer::VisualizerTrait;

#[allow(unused)]
fn tensor_creation() {
    let a = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4], true);
    println!("a is: {:?}", a);
}

#[allow(unused)]
fn test_grad() {
    let x1 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();
    let z = x1.max(Axis(0));

    z.backward(Tensor::ones_like(&z));

    x1.display_grad();
}

fn main() {
    test_grad();
}
