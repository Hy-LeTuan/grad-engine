pub mod config;
pub mod graph;
#[allow(unused)]
use crate::graph::visualizer::VisualizerTrait;
#[allow(unused)]
use crate::ops::public_ops::matmul::matmul;
#[allow(unused)]
use graph::visualizer::Visualizer;
use tensor_core::tensor::Tensor;

// important modules
pub mod ops;
pub mod tensor_core;
pub mod utils;

fn main() {
    let x1 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();
    let x2 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4], true).as_float_32();
    let x3 = Tensor::new(vec![3, 3, 3, 3], vec![4], true).as_float_32();

    let z = matmul(&(&x1 + 3.0), &(&x2 - &x3));

    z.backward(Tensor::ones_like(&z), true);
    Visualizer::visualize_graph(&z);

    x1.display_grad();
    x2.display_grad();
    x3.display_grad();
}
