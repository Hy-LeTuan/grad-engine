use ndarray::ScalarOperand;
use num_traits::Float;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::add_compute::add_compute_tensor_tensor;
use crate::ops::compute::mul_compute::mul_compute_tensor_scalar;
use crate::ops::compute::mul_compute::mul_compute_tensor_tensor;
use crate::ops::compute::pow_compute::pow_compute_tensorimpl;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::ops::Mul;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct PowBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    scalar: Option<T>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for PowBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T> + Mul<Output = T> + Float + ScalarOperand,
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
            if let Some(scalar) = self.scalar.clone() {
                let self_tensor = Rc::clone(&self.input_refs[0]);
                let self_tensor = pow_compute_tensorimpl(self_tensor.deref(), scalar - T::one());
                let self_tensor = mul_compute_tensor_scalar(&self_tensor, scalar);

                let tensor = mul_compute_tensor_tensor(&self_tensor, upstream_gradient);

                return Rc::new(tensor);
            } else {
                panic!("Error, no scalar found on a exp operation of base different than base e");
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

    fn get_origin(&self) -> Option<Rc<RefCell<TensorImpl<T>>>> {
        match &self.origin {
            Some(origin_weak) => {
                let origin_rc = origin_weak.upgrade();
                return origin_rc;
            }
            None => {
                panic!("no origin found on this tensor");
            }
        }
    }
}

impl<T> PowBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = PowBackward {
            name: BackwardType::PowBackward,
            input_refs: vec![],
            id,
            edge_list,
            scalar: None,
            origin: Some(Rc::downgrade(origin)),
        };

        return node;
    }

    pub fn save_scalar(&mut self, scalar: T) {
        self.scalar = Some(scalar);
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;
    use crate::utils::testing_utils::total_test_for_backward_operation;

    #[test]
    fn pow_backward_operation() {
        let x1 = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let z = x1.pow(2.0);

        total_test_for_backward_operation(
            vec![&x1],
            vec![Tensor::new(vec![2, 4, 6, 8], vec![4, 1], false).as_float_32()],
            &z,
            "PowBackward",
            Tensor::new(vec![1, 4, 9, 16], vec![4, 1], false).as_float_32(),
        );
    }
}
