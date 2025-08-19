use super::dtypes::DTComp;
use super::tensor::Tensor;
use super::tensor_impl::TensorImpl;

use super::super::graph::backward::Backward;
use super::super::graph::backward::grad_accum::GradAccum;

use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::ops::{Add, Deref};
use std::rc::Rc;

#[derive(Debug)]
pub struct AutogradMeta<T>
where
    T: DTComp + Debug,
{
    pub name: String,
    pub grad: RefCell<Option<Rc<Tensor<T>>>>,
    pub grad_fn: Option<Rc<RefCell<dyn Backward<T>>>>,
    pub grad_accum: Option<Rc<RefCell<GradAccum<T>>>>,
    pub requires_grad: bool,
    pub is_leaf: bool,
}

impl<T> AutogradMeta<T>
where
    T: DTComp + Debug,
{
    pub fn new_for_intermediate(name: &str) -> Self {
        let autograd_meta = AutogradMeta {
            name: String::from(name),
            grad: RefCell::new(None),
            requires_grad: true,
            grad_fn: None,
            grad_accum: None,
            is_leaf: false,
        };

        return autograd_meta;
    }

    pub fn new_for_leaf(name: String, origin: Rc<RefCell<TensorImpl<T>>>) -> Self {
        let grad_accum = Rc::new(RefCell::new(GradAccum::<T>::new_with_origin(
            vec![],
            origin,
        )));

        let autograd_meta = AutogradMeta {
            name: name,
            grad: RefCell::new(None),
            requires_grad: true,
            grad_fn: None,
            grad_accum: Some(Rc::clone(&grad_accum)),
            is_leaf: true,
        };

        return autograd_meta;
    }

    pub fn set_grad_fn_to_node(&mut self, node: Rc<RefCell<dyn Backward<T>>>) {
        self.grad_fn = Some(Rc::clone(&node));
    }

    pub fn set_grad_accum_to_accum(&mut self, grad_accum_node: Rc<RefCell<GradAccum<T>>>) {
        self.grad_accum = Some(Rc::clone(&grad_accum_node));
    }

    pub fn get_grad_as_ref(&self) -> &RefCell<Option<Rc<Tensor<T>>>> {
        return &self.grad;
    }
    pub fn get_grad_accum(&self) -> &Option<Rc<RefCell<GradAccum<T>>>> {
        return &self.grad_accum;
    }

    pub fn get_grad_fn(&self) -> &Option<Rc<RefCell<dyn Backward<T>>>> {
        return &self.grad_fn;
    }

    pub fn is_leaf(&self) -> bool {
        return self.is_leaf;
    }
}

impl<T> AutogradMeta<T>
where
    T: DTComp + Clone + Debug,
{
    /// Get the actual grad behind the tensor. Must check for existence of grad first before
    /// caling
    pub fn get_grad_as_tensor(&self) -> Rc<Tensor<T>> {
        let ref_to_rc = Ref::map(self.grad.borrow(), |grad_ref| {
            grad_ref.as_ref().expect(
                "Error, attempting to get gradient of a tensor even though the gradient is None.",
            )
        });

        return Rc::clone(ref_to_rc.deref());
    }

    pub fn grad_is_set(&self) -> bool {
        return self.grad.borrow().is_some();
    }

    pub fn set_grad(&self, grad: Rc<Tensor<T>>) {
        let mut tensor_grad = self.grad.borrow_mut();
        if let Some(_existing_grad) = tensor_grad.as_ref() {
            *tensor_grad = Some(grad);
        } else {
            *tensor_grad = Some(grad);
        }
    }
}

impl<T> AutogradMeta<T>
where
    T: DTComp + Clone + Debug + Add<Output = T>,
{
    pub fn start_backprop_chain(&self, starting_gradient: Rc<Tensor<T>>, retain_graph: bool) {
        if self.is_leaf() {
            println!(
                "Warning: Calling backward on leaf tensor will directly set the gradient of the tensor to the starting gradient of backpropagation"
            );
            if let Some(node_arc_ref) = self.get_grad_accum() {
                node_arc_ref.borrow().apply(starting_gradient, retain_graph);
                return;
            }
        } else {
            if let Some(node_arc_ref) = self.get_grad_fn() {
                node_arc_ref.borrow().apply(starting_gradient, retain_graph);
            } else {
                panic!(
                    "Warning: Calling backward on a tensor that is not a leaf tensor and not an intermediate tensor. This tensor has no connection to the computation graph."
                );
            }
        }
    }
}
