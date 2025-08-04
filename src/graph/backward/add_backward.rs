use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct AddBackward<T>
where
    T: DTComp + Clone + Debug,
{
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    #[allow(unused)]
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for AddBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    fn save_grad_to_origin_tensor(&self, _grad: &Rc<Tensor<T>>) {
        return;
    }

    fn apply(&self, upstream_gradient: Rc<Tensor<T>>) {
        // self.save_grad_to_origin_tensor(&upstream_gradient);

        for (_i, edge) in self.get_edge_list().iter().enumerate() {
            let next_grad = self.calculate_gradient_for_next_node(&upstream_gradient, None);

            let next_node = edge.get_next_grad_fn();
            next_node.borrow().apply(next_grad);
        }
    }

    fn calculate_gradient_for_next_node(
        &self,
        upstream_gradient: &Rc<Tensor<T>>,
        _edge: Option<&Edge<T>>,
    ) -> Rc<Tensor<T>> {
        return Rc::clone(upstream_gradient);
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

impl<T> AddBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = AddBackward {
            name: BackwardType::AddBackward,
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
    fn add_backward_creation() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true);
        let c = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true);

        let d = &a + &b + &c;
        let e = &d + 3;

        if e.does_require_grad() {
            assert_eq!(
                e.get_grad_fn().borrow().get_name(),
                String::from("AddBackward"),
                "AddBackward does not exist on tensor from add operation"
            );
        }
    }
}
