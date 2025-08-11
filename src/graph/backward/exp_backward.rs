use num_traits::Float;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::exp_compute::exp_compute_tensorimpl;
use crate::ops::compute::exp_compute::exp2_compute_tensorimpl;
use crate::ops::compute::mul_compute::mul_compute_tensor_tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct ExpBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    natural: bool,
    #[allow(unused)]
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for ExpBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T> + Float,
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
            let input = Rc::clone(&self.input_refs[0]);
            if self.natural {
                let input = exp_compute_tensorimpl(input.deref());
                let result_tensor = mul_compute_tensor_tensor(upstream_gradient.deref(), &input);

                return Rc::new(result_tensor);
            } else {
                let input = exp2_compute_tensorimpl(input.deref());
                let result_tensor = mul_compute_tensor_tensor(upstream_gradient.deref(), &input);

                return Rc::new(result_tensor);
            }
        } else {
            panic!(
                "Error: No edge found to connect to and calculate gradient because exponent is an operation requiring one or more tensor"
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

impl<T> ExpBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = ExpBackward {
            name: BackwardType::ExpBackward,
            input_refs: vec![],
            id,
            edge_list,
            natural: false,
            origin: Some(Rc::downgrade(origin)),
        };

        return node;
    }

    pub fn save_natural_state(&mut self, natural: bool) {
        self.natural = natural;
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;
    use crate::utils::testing_utils::total_test_for_backward_operation;

    #[test]
    fn exp_backward_operation() {
        let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let z = x1.exp();

        total_test_for_backward_operation(
            vec![&x1],
            vec![
                Tensor::new(vec![2.7183, 7.3891, 20.0855, 54.5981], vec![4, 1], false)
                    .as_float_32(),
            ],
            &z,
            "ExpBackward",
            Tensor::new(vec![2.7183, 7.3891, 20.0855, 54.5981], vec![4, 1], false).as_float_32(),
        );
    }
}
