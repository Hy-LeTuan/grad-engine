pub mod config;
pub mod graph;
#[allow(unused)]
use crate::graph::visualize::serialize_graph_fn::{
    export_graph_acyclic, serialize_and_export_graph,
};
use crate::graph::visualize::visualizer::VisualizerTrait;
use crate::ops::public_ops::matmul::matmul;
use graph::visualize::visualizer::Visualizer;
use tensor_core::tensor::Tensor;

// important modules
pub mod ops;
pub mod tensor_core;
pub mod utils;

fn main() {
    let x1 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();
    let x2 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4], true).as_float_32();
    let x3 = Tensor::new(vec![3, 3, 3, 3], vec![4], true).as_float_32();

    let x4 = &x1 + 3.0;
    let x5 = &x2 - &x3;

    let x6 = matmul(&x4, &x5) - &x3;
    let x7 = Tensor::ones_like(&x6);

    x7.requires_grad();

    let z = x6.ln() + &x7.exp();

    z.backward(Tensor::ones_like(&z), true);
    Visualizer::visualize_graph(&z);

    x1.display_grad();
    x2.display_grad();
    x3.display_grad();
    x7.display_grad();

    // serialize_and_export_graph(&z);
    export_graph_acyclic(&z);
}
