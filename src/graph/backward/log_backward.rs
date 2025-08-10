use ndarray::ScalarOperand;
use num_traits::Float;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::div_compute::div_compute_tensor_tensor;
use crate::ops::compute::mul_compute::mul_compute_tensorimpl_scalar;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::ops::Div;
use std::ops::Mul;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct LogBackward<T, S>
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

impl<T, S> Backward<T> for LogBackward<T, S>
where
    T: Clone + DTComp + Debug + 'static + Div<Output = T> + Add<Output = T> + Mul<S, Output = T>,
    S: ScalarOperand + Clone + Debug + Float,
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
        if let Some(_) = edge {
            if let Some(scalar) = self.scalar.clone() {
                let self_tensor = Rc::clone(&self.input_refs[0]);
                let self_tensor = mul_compute_tensorimpl_scalar(self_tensor.deref(), scalar.ln());

                let tensor = div_compute_tensor_tensor(upstream_gradient, &self_tensor);

                return Rc::new(tensor);
            } else {
                panic!("Error, no scalar found on a log operation of base different than base e");
            }
        } else {
            panic!(
                "Error, no edge found to connect to and calculate gradient because ln is a self operation"
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

impl<T, S> LogBackward<T, S>
where
    T: Clone + DTComp + Debug,
    S: ScalarOperand + Clone + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = LogBackward {
            name: BackwardType::LogBackward,
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
    fn log_backward_operation() {
        let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let z = x1.log(2.0);

        total_test_for_backward_operation(
            vec![&x1],
            vec![
                Tensor::new(vec![1.4427, 0.7213, 0.4809, 0.3607], vec![4, 1], false).as_float_32(),
            ],
            &z,
            "LogBackward",
            Tensor::new(vec![0.0000, 1.0000, 1.5850, 2.0000], vec![4, 1], false).as_float_32(),
        );
    }
}
