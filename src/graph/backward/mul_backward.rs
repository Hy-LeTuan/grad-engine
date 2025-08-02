use super::DTypeMarker;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::mul_compute::compute_mul_tensor_tensorimpl;
use crate::tensor_core::tensor_impl::TensorImpl;

use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Deref;
use std::ops::Mul;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct MulBackward<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static + Mul<Output = T>,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for MulBackward<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static + Mul<Output = T>,
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

        for edge in self.get_edge_list().iter() {
            let next_grad = self.calculate_gradient_for_next_node(&upstream_gradient, Some(&edge));

            println!("upstream gradient: {:?}", upstream_gradient);
            println!("computed grad: {:?}", next_grad);
            println!("-----------");

            let next_node = edge.get_next_grad_fn();
            next_node.borrow().apply(next_grad);
        }
    }

    fn calculate_gradient_for_next_node(
        &self,
        upstream_gradient: &Rc<Tensor<T>>,
        edge: Option<&Edge<T>>,
    ) -> Rc<Tensor<T>> {
        if let Some(edge) = edge {
            let edge_nr = edge.input_nr;
            let input_tensor = Rc::clone(&self.input_refs[edge_nr]);

            println!("input tensor: {:?}", input_tensor);

            let tensor = compute_mul_tensor_tensorimpl(input_tensor.deref(), upstream_gradient);

            return Rc::new(tensor);
        } else {
            panic!("Cannot calculate gradient because of missing inputs");
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

impl<T> MulBackward<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static + Mul<Output = T>,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = MulBackward {
            name: BackwardType::MulBackward,
            input_refs: vec![],
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
    fn mul_backward_operation() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();
        let c = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();

        let d = &a * &b * &c;
        let e = &d * 3.0;

        if e.does_require_grad() {
            assert_eq!(
                e.get_grad_fn().borrow().get_name(),
                String::from("MulBackward"),
                "MulBackward does not exist on tensor from mul operation"
            );
        }
    }
}
