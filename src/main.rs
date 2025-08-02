pub mod config;
pub mod graph;
pub mod ops;
pub mod tensor_core;

use graph::visualizer::Visualizer;
use tensor_core::tensor::Tensor;

use crate::graph::visualizer::VisualizerTrait;

fn main() {
    let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
    let x2 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
    let x3 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);

    let z = &x1 + &x2 + &x3;

    Visualizer::visualize_graph(&z);
}
