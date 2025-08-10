use ndarray::ScalarOperand;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::mul_compute::mul_compute_tensor_scalar;
use crate::ops::compute::mul_compute::mul_compute_tensorimpl_tensorimpl;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::ops::Mul;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct MulBackward<T, S>
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

impl<T, S> Backward<T> for MulBackward<T, S>
where
    T: Clone + DTComp + Debug + 'static + Mul<Output = T> + Mul<S, Output = T> + Add<Output = T>,
    S: ScalarOperand + Clone + Debug,
{
    fn save_grad_to_origin_tensor(&self, _grad: &Rc<Tensor<T>>) {
        return;
    }

    fn apply(&self, upstream_gradient: Rc<Tensor<T>>) {
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
        if self.input_refs.len() >= 2 {
            if let Some(edge) = edge {
                let edge_nr = edge.input_nr;

                let tensor;

                if edge_nr == 0 {
                    let input_tensor = Rc::clone(&self.input_refs[1]);
                    tensor = mul_compute_tensorimpl_tensorimpl(
                        input_tensor.deref(),
                        upstream_gradient.__get_tensor_impl(),
                    );
                } else {
                    let input_tensor = Rc::clone(&self.input_refs[0]);
                    tensor = mul_compute_tensorimpl_tensorimpl(
                        input_tensor.deref(),
                        upstream_gradient.__get_tensor_impl(),
                    );
                }

                return Rc::new(tensor);
            } else {
                panic!("Cannot calculate gradient because of missing inputs");
            }
        } else {
            let tensor = mul_compute_tensor_scalar(
                upstream_gradient.deref(),
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

impl<T, S> MulBackward<T, S>
where
    T: Clone + DTComp + Debug,
    S: ScalarOperand + Clone + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = MulBackward {
            name: BackwardType::MulBackward,
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
    fn mul_backward_creation() {
        let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let x2 = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();
        let x3 = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();

        let x4 = &x1 * &x2 * &x3;
        let z = &x4 * 3.0;

        total_test_for_backward_operation(
            vec![&x1, &x2, &x3],
            vec![
                Tensor::new(vec![75, 108, 147, 192], vec![4, 1], false).as_float_32(),
                Tensor::new(vec![15, 36, 63, 96], vec![4, 1], false).as_float_32(),
                Tensor::new(vec![15, 36, 63, 96], vec![4, 1], false).as_float_32(),
            ],
            &z,
            "MulBackward",
            Tensor::new(vec![75, 216, 441, 768], vec![4, 1], false).as_float_32(),
        );
    }
}
