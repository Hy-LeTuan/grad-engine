use grad_engine::cli::cli_utils::cli_args_for_export;
#[allow(unused)]
use grad_engine::graph::visualize::serialize_graph_fn::{
    export_graph_acyclic, serialize_and_export_graph,
};
use grad_engine::graph::visualize::visualizer::Visualizer;
use grad_engine::graph::visualize::visualizer::VisualizerTrait;
use grad_engine::ops::public_ops::matmul::matmul;
use grad_engine::tensor;
use grad_engine::tensor_core::tensor::Tensor;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let x1 = tensor!([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]; requires_grad=true);
    let x2 = tensor!([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]; requires_grad=true);
    let x3 = tensor!(3.0, 3.0, 3.0, 3.0; requires_grad=true);

    let x4 = &x1 + 3.0;
    let x5 = &x2 - &x3;

    let x6 = matmul(&x4, &x5) - &x3;
    let x7 = Tensor::ones_like(&x6, Some(true));

    // x7.requires_grad();

    let z = x6.ln() + &x7.exp();

    z.backward(Tensor::ones_like(&z, None), true);
    Visualizer::visualize_graph(&z);

    x1.display_grad();
    x2.display_grad();
    x3.display_grad();
    x7.display_grad();

    let is_export = cli_args_for_export(args);
    if is_export {
        export_graph_acyclic(&z, None);
    }
}
