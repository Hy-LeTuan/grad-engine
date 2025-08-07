pub mod config;
pub mod graph;
pub mod ops;
pub mod tensor_core;

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
    // let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
    // let x2 = Tensor::new(vec![4, 5, 6, 7], vec![4, 1], true).as_float_32();
    let x3 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();

    let z = x3.tanh();

    z.backward(Tensor::new(vec![1, 1, 1, 1], vec![4, 1], false).as_float_32());

    // x1.display_grad();
    // println!("------");
    //
    // x2.display_grad();
    // println!("------");

    x3.display_grad();
    println!("------");

    // Visualizer::visualize_graph(&z);
}

fn main() {
    tensor_creation();
}
