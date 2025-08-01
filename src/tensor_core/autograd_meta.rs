use super::dtypes::DTypeMarker;
use super::tensor::Tensor;
use super::tensor_impl::TensorImpl;

use super::super::graph::backward::Backward;
use super::super::graph::backward::grad_accum::GradAccum;

use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

#[derive(Debug)]
pub struct AutogradMeta<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static,
{
    pub name: String,
    pub grad: RefCell<Option<Rc<Tensor<T>>>>,
    pub requires_grad: bool,
    pub grad_fn: Option<Rc<RefCell<dyn Backward<T>>>>,
    pub grad_accum: Option<Rc<RefCell<GradAccum<T>>>>,
}

impl<T> AutogradMeta<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static,
{
    pub fn new_for_intermediate(name: &str) -> Self {
        let autograd_meta = AutogradMeta {
            name: String::from(name),
            grad: RefCell::new(None),
            requires_grad: true,
            grad_fn: None,
            grad_accum: None,
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
        };

        return autograd_meta;
    }

    pub fn is_leaf(&self) -> bool {
        if self.grad_fn.is_some() {
            return false;
        } else if self.grad_accum.is_some() {
            return true;
        } else {
            return false;
        }
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

    pub fn set_grad(&self, grad: Rc<Tensor<T>>) {
        let mut tensor_grad = self.grad.borrow_mut();
        if let Some(_existing_grad) = tensor_grad.as_ref() {
            *tensor_grad = Some(grad);
        } else {
            *tensor_grad = Some(grad);
        }
    }

    pub fn get_grad_accum(&self) -> &Option<Rc<RefCell<GradAccum<T>>>> {
        return &self.grad_accum;
    }

    pub fn get_grad_fn(&self) -> &Option<Rc<RefCell<dyn Backward<T>>>> {
        return &self.grad_fn;
    }

    pub fn start_backprop_chain(&self, _starting_gradient: Vec<Tensor<T>>) {
        todo!()
        // let starting_gradient_arc = Rc::new(starting_gradient);
        //
        // if let Some(node_arc_ref) = self.get_grad_accum() {
        //     node_arc_ref.borrow().apply(starting_gradient_arc);
        //     return;
        // }
        //
        // if let Some(node_arc_ref) = self.get_grad_fn() {
        //     node_arc_ref.borrow().apply(starting_gradient_arc);
        // } else {
        //     panic!("Error, calling backwards on a non-leaf tensor with no preceeding operation");
        // }
    }
}
