use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::rc::Rc;

use ndarray::Axis;

use crate::graph::backward::Backward;
use crate::graph::backward::stack_backward::StackBackward;
use crate::graph::edge::Edge;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn stack_impl<T>(tensor_list: &[&Tensor<T>], result_tensor: &Tensor<T>, dim: Axis)
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    let mut node = StackBackward::new(0, vec![], result_tensor.__get_tensor_impl());

    node.save_dim(dim);

    let mut result_does_require_grad = false;

    for (i, tensor) in tensor_list.iter().enumerate() {
        if tensor.does_require_grad() {
            node.add_to_edge_list(Edge::maybe_create_connect(tensor, i));
            node.save_input_refs(vec![tensor.__clone_ptr_to_tensor_impl()]);

            if !result_does_require_grad {
                result_tensor
                    .requires_grad_intermediate("Intermediate tensor from stack operation");
                result_does_require_grad = true;
            }
        }
    }

    if result_does_require_grad {
        let node = Rc::new(RefCell::new(node));
        result_tensor.set_grad_fn(node);
    }
}
