use super::DTypeMarker;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::mul_compute::compute_mul_tensor_scalar;
use crate::tensor_core::tensor_impl::TensorImpl;

use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Deref, Mul};
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct SubBackward<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static,
{
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for SubBackward<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static + Mul<f32, Output = T>,
{
    fn save_grad_to_origin_tensor(&self, grad: &Rc<Tensor<T>>) {
        if let Some(origin_as_option_ref) = self.origin.as_ref() {
            if let Some(origin_as_strong_rc) = origin_as_option_ref.upgrade() {
                if let Some(origin_ref) = origin_as_strong_rc.borrow().get_autograd_ref_().as_ref()
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

        let minuend_grad = self.calculate_gradient_for_next_node(&upstream_gradient);
        let subtrahend_grad = Rc::new(compute_mul_tensor_scalar(
            self.calculate_gradient_for_next_node(&upstream_gradient)
                .deref(),
            -1.0,
        ));

        let minuend_node = &self.get_edge_list()[0].get_next_grad_fn();
        let subtrahend_node = &self.get_edge_list()[1].get_next_grad_fn();

        minuend_node.borrow().apply(minuend_grad);
        subtrahend_node.borrow().apply(subtrahend_grad);
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

    fn get_name(&self) -> String {
        return self.name.to_string();
    }
}

impl<T> SubBackward<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = SubBackward {
            name: BackwardType::SubBackward,
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
        };

        return node;
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn sub_backward_creation() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();
        let c = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();

        let d = &a - &b - &c;
        let e = &d - 3.0;

        if e.does_require_grad() {
            assert_eq!(
                e.get_grad_fn().borrow().get_name(),
                String::from("SubBackward"),
                "SubBackward does not exist on tensor from sub operation"
            );
        }
    }
}
