pub mod config;
pub mod graph;
pub mod ops;
pub mod tensor_core;

use tensor_core::tensor::Tensor;

// use graph::visualizer::Visualizer;
// use crate::graph::visualizer::VisualizerTrait;

fn main() {
    let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
    // let x2 = Tensor::new(vec![4, 5, 6, 7], vec![4, 1], true);
    // let x3 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);

    let z = &x1 + 3;

    z.backward(Tensor::new(vec![1, 1, 1, 1], vec![4, 1], false));

    // Visualizer::visualize_graph(&z);
}
