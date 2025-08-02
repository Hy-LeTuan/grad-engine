use crate::{
    graph::backward::Backward,
    tensor_core::{dtypes::DTypeMarker, tensor::Tensor},
};
use std::collections::VecDeque;
use std::ops::Deref;

use num_traits::Zero;
use std::{cell::RefCell, fmt::Debug, rc::Rc};

pub struct Visualizer {}

pub trait VisualizerTrait<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
{
    fn visualize_graph(tensor: &Tensor<T>);

    fn visualize(node: Rc<RefCell<dyn Backward<T>>>);
}

impl<T> VisualizerTrait<T> for Visualizer
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
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
            Visualizer::visualize(Rc::clone(root));
        }
    }

    fn visualize(root: Rc<RefCell<dyn Backward<T>>>) {
        let mut node_dequeue: VecDeque<(usize, Rc<RefCell<dyn Backward<T>>>)> = VecDeque::new();

        node_dequeue.push_back((0, root));

        while !node_dequeue.is_empty() {
            if let Some((level, node)) = node_dequeue.pop_front() {
                println!(
                    "{:<indent$}-- {:?}",
                    "",
                    node.deref().borrow(),
                    indent = level * 8
                );

                // add to queue
                for edge in node.borrow().get_edge_list() {
                    let next_node = edge.get_next_grad_fn();

                    node_dequeue.push_back((level + 1, next_node));
                }
            }
        }
    }
}
