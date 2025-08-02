pub mod config;
pub mod graph;
pub mod ops;
pub mod tensor_core;

use tensor_core::tensor::Tensor;

// use graph::visualizer::Visualizer;
// use crate::graph::visualizer::VisualizerTrait;

fn main() {
    let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
    let x2 = Tensor::new(vec![4, 5, 6, 7], vec![4, 1], true).as_float_32();
    let x3 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();

    let z = &x1 + &x2 - &x3;
    z.backward(Tensor::new(vec![1, 1, 1, 1], vec![4, 1], false).as_float_32());

    // Visualizer::visualize_graph(&z);
}
