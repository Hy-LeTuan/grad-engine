use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::div_compute::div_compute_tensorimpl_tensorimpl;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::ops::Div;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct LnBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    #[allow(unused)]
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for LnBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Div<Output = T> + Add<Output = T>,
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
            let self_tensor = Rc::clone(&self.input_refs[0]);
            let tensor = div_compute_tensorimpl_tensorimpl(
                upstream_gradient.__get_tensor_impl(),
                self_tensor.deref(),
            );

            return Rc::new(tensor);
        } else {
            panic!(
                "No edge found to connect to and calculate gradient because ln is a self operation"
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

impl<T> LnBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = LnBackward {
            name: BackwardType::LnBackward,
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
    fn ln_backward_operation() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let z = a.ln();

        if z.does_require_grad() {
            assert_eq!(
                z.get_grad_fn().borrow().get_name(),
                String::from("LnBackward"),
                "LnBackward does not exist on tensor from ln operation"
            );
        }
    }
}
