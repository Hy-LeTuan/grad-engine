use super::dtypes::DTypeMarker;
use super::tensor::Tensor;

use super::super::graph::node::Backward;
use super::super::graph::node::grad_accum::GradAccum;

use num_traits::Zero;
use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug)]
pub struct AutogradMeta<T>
where
    T: DTypeMarker + Zero + Clone,
{
    pub name: String,
    pub grad: RefCell<Option<Arc<Tensor<T>>>>,
    pub requires_grad: bool,
    pub grad_fn: Option<Arc<dyn Backward<T>>>,
    pub grad_accum: Option<Arc<GradAccum<T>>>,
}

impl<T> AutogradMeta<T>
where
    T: Zero + Clone + DTypeMarker,
{
    pub fn new(name: String) -> Self {
        let autograd_meta = AutogradMeta {
            name: name,
            grad: RefCell::new(None),
            requires_grad: true,
            grad_fn: None,
            grad_accum: None,
        };

        return autograd_meta;
    }

    pub fn set_grad_fn_to_node(&mut self, node: Arc<dyn Backward<T>>) {
        self.grad_fn = Some(Arc::clone(&node));
    }

    pub fn set_grad_accum_to_accum(&mut self, grad_accum_node: Arc<GradAccum<T>>) {
        self.grad_accum = Some(Arc::clone(&grad_accum_node));
    }

    pub fn set_grad(&self, grad: Arc<Tensor<T>>) {
        let mut tensor_grad = self.grad.borrow_mut();
        if let Some(_existing_grad) = tensor_grad.as_ref() {
            *tensor_grad = Some(grad);
        } else {
            *tensor_grad = Some(grad);
        }
    }
}
