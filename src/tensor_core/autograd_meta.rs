use super::dtypes::DTypeMarker;
use super::tensor::Tensor;

use super::super::graph::node::Backward;
use super::super::graph::node::grad_accum::GradAccum;

use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct AutogradMeta<T>
where
    T: DTypeMarker + Zero + Clone + Debug,
{
    pub name: String,
    pub grad: RefCell<Option<Arc<Tensor<T>>>>,
    pub requires_grad: bool,
    pub grad_fn: Option<Arc<RefCell<dyn Backward<T>>>>,
    pub grad_accum: Option<Arc<RefCell<GradAccum<T>>>>,
}

impl<T> AutogradMeta<T>
where
    T: DTypeMarker + Zero + Clone + Debug,
{
    pub fn new_for_intermediate(
        name: String,
        grad_fn: Option<Arc<RefCell<dyn Backward<T>>>>,
    ) -> Self {
        let autograd_meta = AutogradMeta {
            name: name,
            grad: RefCell::new(None),
            requires_grad: true,
            grad_fn: grad_fn,
            grad_accum: None,
        };

        return autograd_meta;
    }

    pub fn new_for_leaf(name: String) -> Arc<Self> {
        let grad_accum = Arc::new(RefCell::new(GradAccum::<T>::new(vec![], vec![])));

        let autograd_meta = AutogradMeta {
            name: name,
            grad: RefCell::new(None),
            requires_grad: true,
            grad_fn: None,
            grad_accum: Some(Arc::clone(&grad_accum)),
        };

        let autograd_meta_arc = Arc::new(autograd_meta);

        grad_accum
            .borrow_mut()
            .set_owned_meta(Arc::clone(&autograd_meta_arc));

        return autograd_meta_arc;
    }

    pub fn set_grad_fn_to_node(&mut self, node: Arc<RefCell<dyn Backward<T>>>) {
        self.grad_fn = Some(Arc::clone(&node));
    }

    pub fn set_grad_accum_to_accum(&mut self, grad_accum_node: Arc<RefCell<GradAccum<T>>>) {
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
