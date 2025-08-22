use grad_engine::cli::cli_utils::cli_args_for_export;
#[allow(unused)]
use grad_engine::graph::visualize::serialize_graph_fn::{
    export_graph_acyclic, serialize_and_export_graph,
};
use grad_engine::graph::visualize::visualizer::Visualizer;
use grad_engine::graph::visualize::visualizer::VisualizerTrait;
use grad_engine::tensor;
use grad_engine::tensor_core::tensor::Tensor;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let x1 = tensor!(1.0; requires_grad=true);
    let x2 = tensor!(2.0; requires_grad=true);
    let x3 = tensor!(3.0; requires_grad=true);

    let x4 = x1.pow(2.0) + (&x2 * &x3);
    let x5 = &x1 + &x2.pow(2.0);

    let z = &x4 + &x5;

    z.backward(Tensor::ones_like(&z, None), true);
    Visualizer::visualize_graph(&z);

    x1.display_grad();
    x2.display_grad();
    x3.display_grad();

    let is_export = cli_args_for_export(args);
    if is_export {
        export_graph_acyclic(&z, None);
    }
}
