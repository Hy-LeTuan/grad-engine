pub mod config;
pub mod graph;
pub mod ops;
pub mod tensor_core;
pub mod utils;

use tensor_core::tensor::Tensor;

use crate::ops::public_ops::matmul::matmul;

use crate::graph::visualizer::VisualizerTrait;
use graph::visualizer::Visualizer;

#[allow(unused)]
fn tensor_creation() {
    let a = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4], true);
    println!("a is: {:?}", a);
}

#[allow(unused)]
fn test_grad() {
    let x1 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();
    let x2 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4], true).as_float_32();

    let z = matmul(&(&x1 + 3.0), &x2);

    z.backward(Tensor::ones_like(&z));

    Visualizer::visualize_graph(&z);
}

fn main() {
    test_grad();
}
