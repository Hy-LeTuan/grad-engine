#[allow(unused)]
use grad_engine::graph::visualize::serialize_graph_fn::{
    export_graph_acyclic, serialize_and_export_graph,
};
use grad_engine::graph::visualize::visualizer::Visualizer;
use grad_engine::graph::visualize::visualizer::VisualizerTrait;
use grad_engine::tensor_core::tensor::Tensor;

fn main() {
    let x1 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();
    let x2 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![2, 4], true).as_float_32();
    let x3 = Tensor::new(vec![3, 3, 3, 3], vec![4], true).as_float_32();

    let z = &x1 + &x2 - &x3;

    z.backward(Tensor::ones_like(&z), true);
    Visualizer::visualize_graph(&z);

    x1.display_grad();
    x2.display_grad();
    x3.display_grad();
}
