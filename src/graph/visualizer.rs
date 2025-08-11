use crate::{
    graph::backward::Backward,
    tensor_core::{dtypes::DTComp, tensor::Tensor},
};
use colored::*;

use num_traits::Zero;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

pub struct Visualizer {}

pub trait VisualizerTrait<T>
where
    T: Zero + Clone + DTComp + Debug + 'static,
{
    fn visualize_graph(tensor: &Tensor<T>);
    fn visualize_node_dfs(node: Rc<RefCell<dyn Backward<T>>>, level: usize, is_last: bool);
}

impl<T> VisualizerTrait<T> for Visualizer
where
    T: Zero + Clone + DTComp + Debug + 'static,
{
    fn visualize_graph(tensor: &Tensor<T>) {
        if !tensor.does_require_grad() {
            return;
        }

        let autograd_ref = tensor.get_autograd_ref();
        let root = autograd_ref
            .as_ref()
            .expect("Autograd does not exist")
            .get_grad_fn();

        if let Some(root) = root.as_ref() {
            println!("## Backward computation graph ##");
            println!("------------------------------");
            println!("");
            Visualizer::visualize_node_dfs(Rc::clone(root), 0, false);
            println!("");
            println!("------------------------------");
        }
    }

    fn visualize_node_dfs(node: Rc<RefCell<dyn Backward<T>>>, level: usize, is_last: bool) {
        let borrowed = node.borrow();
        let edges = borrowed.get_edge_list();
        let edge_count = edges.len();

        // Tree connector symbols
        let connector = if level == 0 {
            "".to_string()
        } else if is_last {
            format!("{:indent$}└── ", "", indent = (level - 1) * 4)
        } else {
            format!("{:indent$}├── ", "", indent = (level - 1) * 4)
        };

        // Node label with color
        let label = format!("{}", borrowed);

        if label == "GradAccum" {
            println!(
                "{}{} {}",
                connector,
                label.green(),
                ("[ Leaf grad accumulation ]").yellow()
            );
        } else {
            println!(
                "{}{} {}",
                connector,
                label.green(),
                format!("[ {} child nodes ]", edge_count).blue()
            );
        }

        // Recurse into children
        for (i, edge) in edges.into_iter().enumerate() {
            let next_node = edge.get_next_grad_fn();
            Visualizer::visualize_node_dfs(next_node, level + 1, i == edge_count - 1);
        }
    }
}
