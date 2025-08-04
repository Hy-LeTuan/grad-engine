use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::mul_compute::mul_compute_reverse_tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

use num_traits::Signed;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct SubBackward<T>
where
    T: DTComp + Clone + Debug + 'static,
{
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    #[allow(unused)]
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for SubBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Signed,
{
    fn save_grad_to_origin_tensor(&self, _grad: &Rc<Tensor<T>>) {
        return;
    }

    fn apply(&self, upstream_gradient: Rc<Tensor<T>>) {
        for edge in self.get_edge_list().iter() {
            let grad = self.calculate_gradient_for_next_node(&upstream_gradient, Some(edge));

            let node = edge.get_next_grad_fn();
            node.borrow().apply(grad);
        }
    }

    fn calculate_gradient_for_next_node(
        &self,
        upstream_gradient: &Rc<Tensor<T>>,
        edge: Option<&Edge<T>>,
    ) -> Rc<Tensor<T>> {
        if let Some(edge) = edge {
            let input_nr = edge.input_nr;

            if input_nr == 0 {
                return Rc::clone(upstream_gradient);
            } else {
                let subtrahend_grad = mul_compute_reverse_tensor(upstream_gradient.deref());
                return Rc::new(subtrahend_grad);
            }
        } else {
            panic!("No edge positinal data found to calculate gradient")
        }
    }

    fn get_edge_list(&self) -> &[Edge<T>] {
        return &self.edge_list;
    }

    fn add_to_edge_list(&mut self, edge: Edge<T>) {
        self.edge_list.push(edge);
    }

    fn save_input_refs(&mut self, _input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>) {
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
    T: Clone + DTComp + Debug,
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
        // let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();
        // let c = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();

        // let d = &a - &b - &c;
        let e = &a - 3.0;

        if e.does_require_grad() {
            assert_eq!(
                e.get_grad_fn().borrow().get_name(),
                String::from("SubBackward"),
                "SubBackward does not exist on tensor from sub operation"
            );

            // e.backward(Tensor::new(vec![1, 1, 1, 1], vec![4, 1], false).as_float_32());
        }
    }
}
