use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Mul;
use std::rc::Rc;

use crate::graph::backward::Backward;
use crate::graph::backward::mul_backward::MulBackward;
use crate::graph::edge::Edge;
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn mul_impl<TensorType>(
    lhs_tensor: Option<&Tensor<TensorType>>,
    rhs_tensor: Option<&Tensor<TensorType>>,
    result_tensor: &Tensor<TensorType>,
) where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType>,
{
    if !result_tensor.does_require_grad() {
        return;
    }

    let mut node = MulBackward::new(0, vec![], result_tensor.__get_tensor_impl());

    match (lhs_tensor, rhs_tensor) {
        (Some(l), Some(r)) => {
            if l.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(l, 0));
                node.save_input_refs(vec![l.__clone_ptr_to_tensor_impl()]);
            }

            if r.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(r, 1));
                node.save_input_refs(vec![r.__clone_ptr_to_tensor_impl()]);
            }
        }
        (Some(l), None) => {
            if l.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(l, 0));
                node.save_input_refs(vec![l.__clone_ptr_to_tensor_impl()]);
            }
        }
        (None, Some(r)) => {
            if r.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(r, 0));
                node.save_input_refs(vec![r.__clone_ptr_to_tensor_impl()]);
            }
        }
        (None, None) => {
            return;
        }
    }

    let node = Rc::new(RefCell::new(node));
    result_tensor.set_grad_fn(node);
}
