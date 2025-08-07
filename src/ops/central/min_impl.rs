use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::rc::Rc;

use ndarray::Axis;
use num_traits::Zero;

use crate::graph::backward::Backward;
use crate::graph::backward::min_backward::MinBackward;
use crate::graph::edge::Edge;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn min_impl<T>(
    lhs_tensor: Option<&Tensor<T>>,
    result_tensor: &Tensor<T>,
    indices: Rc<Tensor<usize>>,
    reduced_dim: Axis,
) where
    T: Clone + DTComp + Debug + 'static + Add<Output = T> + Zero,
{
    if !result_tensor.does_require_grad() {
        return;
    }

    let mut node = MinBackward::new(0, vec![], result_tensor.__get_tensor_impl());

    match lhs_tensor {
        Some(l) => {
            if l.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(l, 0));
            }

            node.save_input_refs(vec![l.__clone_ptr_to_tensor_impl()]);
            node.save_indices(indices);
            node.save_reduced_dim(reduced_dim);
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
