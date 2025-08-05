use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Add, Mul};
use std::rc::Rc;

use ndarray::ScalarOperand;
use num_traits::Float;

use crate::graph::backward::Backward;
use crate::graph::backward::tanh_backward::TanhBackward;
use crate::graph::edge::Edge;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn tanh_impl<T>(lhs_tensor: Option<&Tensor<T>>, result_tensor: &Tensor<T>)
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T> + Mul<Output = T> + Float + ScalarOperand,
{
    if !result_tensor.does_require_grad() {
        return;
    }

    let mut node = TanhBackward::new(0, vec![], result_tensor.__get_tensor_impl());

    match lhs_tensor {
        Some(l) => {
            if l.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(l, 0));
            }

            node.save_input_refs(vec![l.__clone_ptr_to_tensor_impl()]);
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
