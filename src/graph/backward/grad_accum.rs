use super::super::super::tensor_core::tensor_impl::TensorImpl;
use super::super::backward::Backward;
use super::super::edge::Edge;
use crate::ops::compute::add_compute;

use super::DTypeMarker;
use super::Tensor;

use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
{
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
{
    fn save_grad_to_origin_tensor(&self, grad: &Rc<Tensor<T>>) {
        if let Some(origin_as_option_ref) = self.origin.as_ref() {
            if let Some(origin_as_strong_rc) = origin_as_option_ref.upgrade() {
                if let Some(origin_ref) = origin_as_strong_rc
                    .borrow_mut()
                    .get_autograd_ref_()
                    .as_ref()
                {
                    match origin_ref.get_grad_as_ref().borrow().as_ref() {
                        Some(x) => {
                            origin_ref.set_grad(Rc::new(add_compute::compute_add_tensor_tensor(
                                x.deref(),
                                grad.deref(),
                            )));
                        }
                        None => {
                            origin_ref.set_grad(Rc::clone(grad));
                        }
                    };
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
        let next_grad = self.calculate_gradient_for_next_node(&upstream_gradient);
        self.save_grad_to_origin_tensor(&next_grad);
    }

    fn calculate_gradient_for_next_node(&self, upstream_gradient: &Rc<Tensor<T>>) -> Rc<Tensor<T>> {
        return Rc::clone(upstream_gradient);
    }

    fn get_edge_list(&self) -> &[Edge<T>] {
        return &self.edge_list;
    }

    fn add_to_edge_list(&mut self, _edge: Edge<T>) {
        return;
    }

    fn save_input_refs(&mut self, _input_refs: &[&Tensor<T>]) {
        return;
    }

    fn get_id(&self) -> usize {
        return self.id;
    }
}

impl<T> GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    pub fn new(edge_list: Vec<Edge<T>>) -> Self {
        let grad_accum = GradAccum {
            id: 0,
            edge_list: edge_list,
            origin: None,
        };

        return grad_accum;
    }

    pub fn new_with_origin(edge_list: Vec<Edge<T>>, origin: Rc<RefCell<TensorImpl<T>>>) -> Self {
        let grad_accum = GradAccum {
            id: 0,
            edge_list: edge_list,
            origin: Some(GradAccum::convert_origin_to_weak(origin)),
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
        let _grad_accum = GradAccum::<f32>::new(vec![]);
    }
}
