use ndarray::Axis;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::add_compute::add_compute_tensor_tensor;
use crate::ops::compute::shape_compute::compute_squeeze;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct UnsqueezeBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
    dim: Axis,
}

impl<T> Backward<T> for UnsqueezeBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    fn save_grad_to_origin_tensor(&self, grad: &Rc<Tensor<T>>) {
        if let Some(origin_as_option_ref) = self.origin.as_ref() {
            if let Some(origin_as_strong_rc) = origin_as_option_ref.upgrade() {
                if let Some(origin_ref) = origin_as_strong_rc.borrow().get_autograd_ref_().as_ref()
                {
                    if origin_ref.grad_is_set() {
                        let old_grad = origin_ref.get_grad_as_tensor();
                        let new_grad = add_compute_tensor_tensor(old_grad.deref(), grad.deref());

                        origin_ref.set_grad(Rc::new(new_grad));
                    } else {
                        origin_ref.set_grad(Rc::clone(grad));
                    }
                }
            }
        } else {
            panic!(
                "Dangling graph node, no origin tensor found at node: {} with id: {}",
                self.get_name(),
                self.get_id(),
            );
        }
    }

    fn apply(&self, upstream_gradient: Rc<Tensor<T>>, retain_graph: bool) {
        if retain_graph {
            self.save_grad_to_origin_tensor(&upstream_gradient);
        }

        for edge in self.get_edge_list().iter() {
            let next_grad = self.calculate_gradient_for_next_node(&upstream_gradient, Some(&edge));

            let next_node = edge.get_next_grad_fn();
            next_node.borrow().apply(next_grad, retain_graph);
        }
    }

    fn calculate_gradient_for_next_node(
        &self,
        upstream_gradient: &Rc<Tensor<T>>,
        edge: Option<&Edge<T>>,
    ) -> Rc<Tensor<T>> {
        if let Some(_) = edge {
            let result_tensor = compute_squeeze(upstream_gradient.deref(), Axis(self.dim.index()));
            return Rc::new(result_tensor);
        } else {
            panic!(
                "Error, no edge found to connect to and calculate gradient because unsqueeze is an operation requiring one or more tensor"
            );
        }
    }

    fn get_edge_list(&self) -> &[Edge<T>] {
        return &self.edge_list;
    }

    fn add_to_edge_list(&mut self, edge: Edge<T>) {
        self.edge_list.push(edge);
    }

    fn save_input_refs(&mut self, input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>) {
        self.input_refs.extend(input_refs);
    }

    fn get_id(&self) -> usize {
        return self.id;
    }

    fn get_name(&self) -> String {
        return self.name.to_string();
    }
}

impl<T> UnsqueezeBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = UnsqueezeBackward {
            name: BackwardType::UnsqueezeBackward,
            input_refs: vec![],
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
            dim: Axis(0),
        };

        return node;
    }

    pub fn save_reduced_dim(&mut self, dim: Axis) {
        self.dim = dim;
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;
    use crate::utils::testing_utils::total_test_for_backward_operation;

    #[test]
    fn unsqueeze_backward_operation() {
        let x1 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();
        let z = x1.unsqueeze(Axis(0));

        total_test_for_backward_operation(
            vec![&x1],
            vec![Tensor::new(vec![1, 1, 1, 1, 1, 1, 1, 1], vec![1, 4, 2], false).as_float_32()],
            &z,
            "UnsqueezeBackward",
            Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![1, 4, 2], false).as_float_32(),
        );
    }
}
