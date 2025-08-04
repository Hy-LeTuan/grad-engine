use ndarray::ScalarOperand;
use num_traits::Signed;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::div_compute::div_compute_tensor_tensor;
use crate::ops::compute::div_compute::{
    div_compute_tensor_tensorimpl, div_compute_tensorimpl_scalar,
};
use crate::ops::compute::mul_compute::mul_compute_reverse_tensor;
use crate::ops::compute::mul_compute::mul_compute_tensorimpl_tensorimpl;
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
    #[allow(unused)]
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
    fn save_grad_to_origin_tensor(&self, _grad: &Rc<Tensor<T>>) {
        return;
    }

    fn apply(&self, upstream_gradient: Rc<Tensor<T>>) {
        // self.save_grad_to_origin_tensor(&upstream_gradient);

        for edge in self.get_edge_list().iter() {
            let next_grad = self.calculate_gradient_for_next_node(&upstream_gradient, Some(&edge));

            let next_node = edge.get_next_grad_fn();
            next_node.borrow().apply(next_grad);
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

                if edge_nr == 0 {
                    let other_tensor = Rc::clone(&self.input_refs[1]);
                    tensor = div_compute_tensor_tensorimpl(
                        other_tensor.deref(),
                        upstream_gradient,
                        false,
                    );
                } else {
                    let other_tensor = Rc::clone(&self.input_refs[0]);
                    let self_tensor = Rc::clone(&self.input_refs[1]);

                    let self_tensor =
                        mul_compute_tensorimpl_tensorimpl(self_tensor.deref(), self_tensor.deref());
                    let reverse_upstream_gradient =
                        mul_compute_reverse_tensor(upstream_gradient.deref());

                    let product_tensor = mul_compute_tensorimpl_tensorimpl(
                        other_tensor.deref(),
                        reverse_upstream_gradient.__get_tensor_impl(),
                    );

                    tensor = div_compute_tensor_tensor(&product_tensor, &self_tensor);
                }
                return Rc::new(tensor);
            } else {
                panic!(
                    "Cannot calculate gradient because of wrong number of inputs. Expected 2 inputs, received {} inputs.",
                    self.input_refs.len()
                );
            }
        } else {
            let input_tensor = Rc::clone(&self.input_refs[0]);

            let tensor = div_compute_tensorimpl_scalar(
                input_tensor.deref(),
                self.scalar
                    .as_ref()
                    .expect("Cannot calculate gradient, missing scalar")
                    .clone(),
            );

            return Rc::new(tensor);
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

    #[test]
    fn div_backward_operation() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();
        let c = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();

        let d = &a / &b / &c;
        let e = &d / 3.0;

        if e.does_require_grad() {
            assert_eq!(
                e.get_grad_fn().borrow().get_name(),
                String::from("DivBackward"),
                "DivBackward does not exist on tensor from div operation"
            );
        }
    }
}
