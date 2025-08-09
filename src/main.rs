pub mod config;
pub mod graph;
pub mod ops;
pub mod tensor_core;
pub mod utils;

use tensor_core::tensor::Tensor;

use crate::ops::public_ops::matmul_public::matmul_tensor_tensor;

// use graph::visualizer::Visualizer;
// use crate::graph::visualizer::VisualizerTrait;

#[allow(unused)]
fn tensor_creation() {
    let a = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4], true);
    println!("a is: {:?}", a);
}

#[allow(unused)]
fn test_grad() {
    let x1 = Tensor::new(
        vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        ],
        vec![3, 2, 3],
        true,
    )
    .as_float_32();

    let x2 = Tensor::new(
        vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        ],
        vec![3, 3, 4],
        true,
    )
    .as_float_32();

    let z = matmul_tensor_tensor(&x1, &x2);

    z.backward(
        Tensor::new(
            vec![
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            ],
            vec![3, 2, 4],
            false,
        )
        .as_float_32(),
    );

    println!("\n------\n");
    x1.display_grad();
    println!("\n------\n");
    x2.display_grad();

    // Visualizer::visualize_graph(&z);
}

fn main() {
    test_grad();
}
