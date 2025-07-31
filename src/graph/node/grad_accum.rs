use super::super::edge::Edge;
use super::super::node::Backward;
use super::DTypeMarker;

use super::super::super::tensor_core::tensor_impl::TensorImpl;
use super::Tensor;

use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
    inputs: Vec<Rc<Tensor<T>>>,
}

impl<T> Backward<T> for GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    fn calculate_gradient(&self, _others: Rc<Vec<Tensor<T>>>) -> Rc<Vec<Tensor<T>>> {
        return Rc::new(vec![]);
    }

    fn traverse(&self) {
        todo!()
    }

    fn get_edge_list(&self) -> &[Edge<T>] {
        return &self.edge_list;
    }

    fn save_input_refs(&mut self, _input_refs: &[&Tensor<T>]) {
        todo!()
    }

    /// save gradient to the origin tensor
    fn save_grad_to_origin_tensor(&self, _grad: Rc<Tensor<T>>) {
        todo!()
    }

    fn add_to_edge_list(&mut self, _edge: Edge<T>) {
        todo!()
    }
}

impl<T> GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    pub fn new(edge_list: Vec<Edge<T>>, inputs: Vec<Rc<Tensor<T>>>) -> Self {
        let grad_accum = GradAccum {
            edge_list: edge_list,
            origin: None,
            inputs: inputs,
        };

        return grad_accum;
    }

    pub fn new_with_origin(edge_list: Vec<Edge<T>>, origin: Rc<RefCell<TensorImpl<T>>>) -> Self {
        let grad_accum = GradAccum {
            edge_list: edge_list,
            origin: Some(GradAccum::convert_origin_to_weak(origin)),
            inputs: vec![],
        };

        return grad_accum;
    }

    pub fn convert_origin_to_weak(
        origin: Rc<RefCell<TensorImpl<T>>>,
    ) -> Weak<RefCell<TensorImpl<T>>> {
        return Rc::downgrade(&origin);
    }

    pub fn set_owned_meta(&mut self, origin: Rc<RefCell<TensorImpl<T>>>) {
        self.origin = Some(Rc::downgrade(&origin));
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn grad_accum_create() {
        let _grad_accum = GradAccum::<f32>::new(vec![], vec![]);
    }
}
