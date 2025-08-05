use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul};
use std::rc::Rc;

use ndarray::ScalarOperand;
use num_traits::Float;

use crate::graph::backward::Backward;
use crate::graph::backward::log_backward::LogBackward;
use crate::graph::edge::Edge;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn log_impl<T, S>(lhs_tensor: Option<&Tensor<T>>, result_tensor: &Tensor<T>, scalar: Option<S>)
where
    T: Clone + DTComp + Debug + 'static + Div<Output = T> + Add<Output = T> + Mul<S, Output = T>,
    S: ScalarOperand + Clone + Debug + Float,
{
    if !result_tensor.does_require_grad() {
        return;
    }

    let mut node = LogBackward::new(0, vec![], result_tensor.__get_tensor_impl());

    match lhs_tensor {
        Some(l) => {
            if l.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(l, 0));
            }

            node.save_input_refs(vec![l.__clone_ptr_to_tensor_impl()]);
            node.save_scalar(scalar.expect("Error, no log base found").clone());
        }
        None => {
            panic!(
                "Error, No input found, input is needed to calculate gradient of a ln operation."
            );
        }
    }

    let node = Rc::new(RefCell::new(node));
    result_tensor.set_grad_fn(node);
}
