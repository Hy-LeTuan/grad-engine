use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Div};
use std::rc::Rc;

use ndarray::{Axis, ScalarOperand};
use num_traits::NumCast;

use crate::graph::backward::Backward;
use crate::graph::backward::mean_backward::MeanBackward;
use crate::graph::edge::Edge;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn mean_impl<T>(lhs_tensor: Option<&Tensor<T>>, result_tensor: &Tensor<T>, reduced_dim: Axis)
where
    T: Clone
        + DTComp
        + Debug
        + 'static
        + Add<Output = T>
        + Div<Output = T>
        + NumCast
        + ScalarOperand,
{
    if !result_tensor.does_require_grad() {
        return;
    }

    let mut node = MeanBackward::new(0, vec![], result_tensor.__get_tensor_impl());

    match lhs_tensor {
        Some(l) => {
            if l.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(l, 0));
            }

            node.save_input_refs(vec![l.__clone_ptr_to_tensor_impl()]);
            node.save_reduced_dim(reduced_dim);
        }
        None => {
            panic!(
                "Error, No input found, input is needed to calculate gradient of a sum operation."
            );
        }
    }

    let node = Rc::new(RefCell::new(node));
    result_tensor.set_grad_fn(node);
}
