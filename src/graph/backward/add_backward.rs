use super::DTypeMarker;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::edge::Edge;
use crate::tensor_core::tensor_impl::TensorImpl;

use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct AddBackward<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static,
{
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for AddBackward<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
{
    fn save_grad_to_origin_tensor(&self, grad: &Rc<Tensor<T>>) {
        if let Some(origin_as_option_ref) = self.origin.as_ref() {
            if let Some(origin_as_strong_rc) = origin_as_option_ref.upgrade() {
                if let Some(origin_ref) = origin_as_strong_rc
                    .borrow_mut()
                    .get_autograd_ref_()
                    .as_ref()
                {
                    origin_ref.set_grad(Rc::clone(grad));
                }
            }
        } else {
            panic!(
                "Dangling graph node, no origin tensor found at node: {}",
                self.get_id()
            );
        }
    }

    fn apply(&self, upstream_gradient: Rc<Tensor<T>>) {
        self.save_grad_to_origin_tensor(&upstream_gradient);

        for (_i, edge) in self.get_edge_list().iter().enumerate() {
            let next_grad = self.calculate_gradient_for_next_node(&upstream_gradient);

            let next_node = edge.get_next_grad_fn();
            next_node.borrow().apply(next_grad);
        }
    }

    fn calculate_gradient_for_next_node(&self, upstream_gradient: &Rc<Tensor<T>>) -> Rc<Tensor<T>> {
        return Rc::clone(upstream_gradient);
    }

    fn get_edge_list(&self) -> &[Edge<T>] {
        return &self.edge_list;
    }

    fn add_to_edge_list(&mut self, edge: Edge<T>) {
        self.edge_list.push(edge);
    }

    fn save_input_refs(&mut self, _input_refs: &[&Tensor<T>]) {
        return;
    }

    fn get_id(&self) -> usize {
        return self.id;
    }
}

impl<T> AddBackward<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = AddBackward {
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
        };

        return node;
    }
}
