use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::rc::Rc;

use ndarray::Axis;

use crate::graph::backward::Backward;
use crate::graph::backward::concat_backward::ConcatBackward;
use crate::graph::edge::Edge;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn concat_impl<T>(tensor_list: &[&Tensor<T>], result_tensor: &Tensor<T>, dim: Axis)
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    let mut node = ConcatBackward::new(0, vec![], result_tensor.__get_tensor_impl());
    node.save_dim(dim);

    let mut result_does_require_grad = false;
    let mut accumulated_size: usize = 0;

    for (i, tensor) in tensor_list.iter().enumerate() {
        if tensor.does_require_grad() {
            node.add_to_edge_list(Edge::maybe_create_connect(tensor, i));
            node.save_input_refs(vec![tensor.__clone_ptr_to_tensor_impl()]);

            let range = (
                accumulated_size,
                accumulated_size + tensor.get_shape()[dim.index()],
            );

            node.save_ranges(range);
            accumulated_size += tensor.get_shape()[dim.index()];

            // set result to also require grad if any grad tracking is enabled
            if !result_does_require_grad {
                result_tensor
                    .requires_grad_intermediate("Intermediate tensor from concat operation");
                result_does_require_grad = true;
            }
        }
    }

    if result_does_require_grad {
        let node = Rc::new(RefCell::new(node));
        result_tensor.set_grad_fn(node);
    }
}
