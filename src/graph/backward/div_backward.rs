use ndarray::ScalarOperand;
use num_traits::Signed;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::backward::backward_utils::gradient_from_broadcast;
use crate::graph::edge::Edge;
use crate::ops::compute::add_compute::add_compute_tensor_tensor;
use crate::ops::compute::div_compute::div_compute_tensor_scalar;
use crate::ops::compute::div_compute::div_compute_tensor_tensor;
use crate::ops::compute::div_compute::div_compute_tensorimpl_tensorimpl;
use crate::ops::compute::mul_compute::mul_compute_tensorimpl_tensorimpl;
use crate::ops::compute::neg_compute::neg_compute_tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::ops::Div;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct DivBackward<T, S>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    scalar: Option<S>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T, S> Backward<T> for DivBackward<T, S>
where
    T: Clone
        + DTComp
        + Debug
        + 'static
        + Div<Output = T>
        + Div<S, Output = T>
        + Add<Output = T>
        + Signed,
    S: ScalarOperand + Clone + Debug,
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
        if self.input_refs.len() == 2 {
            if let Some(edge) = edge {
                let edge_nr = edge.input_nr;

                let tensor;
                let intended_shape;

                if edge_nr == 0 {
                    let other_tensor = Rc::clone(&self.input_refs[1]);
                    tensor = div_compute_tensorimpl_tensorimpl(
                        upstream_gradient.__get_tensor_impl(),
                        other_tensor.deref(),
                    );
                    intended_shape = self.input_refs[0].borrow().get_raw_shape();
                } else {
                    let other_tensor = Rc::clone(&self.input_refs[0]);
                    let self_tensor = Rc::clone(&self.input_refs[1]);

                    let self_tensor =
                        mul_compute_tensorimpl_tensorimpl(self_tensor.deref(), self_tensor.deref());
                    let reverse_upstream_gradient = neg_compute_tensor(upstream_gradient.deref());

                    let product_tensor = mul_compute_tensorimpl_tensorimpl(
                        other_tensor.deref(),
                        reverse_upstream_gradient.__get_tensor_impl(),
                    );

                    tensor = div_compute_tensor_tensor(&product_tensor, &self_tensor);
                    intended_shape = self.input_refs[1].borrow().get_raw_shape();
                }
                return Rc::new(gradient_from_broadcast(&tensor, &intended_shape));
            } else {
                panic!(
                    "Cannot calculate gradient because of wrong number of inputs. Expected 2 inputs, received {} inputs.",
                    self.input_refs.len()
                );
            }
        } else {
            let tensor = div_compute_tensor_scalar(
                upstream_gradient.deref(),
                self.scalar
                    .as_ref()
                    .expect("Cannot calculate gradient, missing scalar")
                    .clone(),
            );

            return Rc::new(gradient_from_broadcast(
                &tensor,
                &self.input_refs[0].borrow().get_raw_shape(),
            ));
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

impl<T, S> DivBackward<T, S>
where
    T: Clone + DTComp + Debug,
    S: ScalarOperand + Clone + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = DivBackward {
            name: BackwardType::DivBackward,
            input_refs: vec![],
            id,
            edge_list,
            scalar: None,
            origin: Some(Rc::downgrade(origin)),
        };

        return node;
    }

    pub fn save_scalar(&mut self, scalar: S) {
        self.scalar = Some(scalar);
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;
    use crate::utils::testing_utils::total_test_for_backward_operation;

    #[test]
    fn div_backward_operation() {
        let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let x2 = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();
        let x3 = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();

        let x4 = &x1 / &x2 / &x3;
        let z = &x4 / 3.0;

        total_test_for_backward_operation(
            vec![&x1, &x2, &x3],
            vec![
                Tensor::new(vec![0.0133, 0.0093, 0.0068, 0.0052], vec![4, 1], false).as_float_32(),
                Tensor::new(vec![-0.0027, -0.0031, -0.0029, -0.0026], vec![4, 1], false)
                    .as_float_32(),
                Tensor::new(vec![-0.0027, -0.0031, -0.0029, -0.0026], vec![4, 1], false)
                    .as_float_32(),
            ],
            &z,
            "DivBackward",
            Tensor::new(vec![0.0133, 0.0185, 0.0204, 0.0208], vec![4, 1], false).as_float_32(),
        );
    }
}
