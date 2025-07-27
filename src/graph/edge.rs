use super::node::Backward;

use super::super::tensor_core::dtypes::DTypeMarker;
use super::super::tensor_core::tensor::Tensor;

use num_traits::Zero;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct Edge<T>
where
    T: DTypeMarker + Zero + Clone,
{
    pub grad_fn_linked: Arc<dyn Backward<T>>,
    pub input_nr: usize,
}

impl<T> Edge<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static,
{
    /// Connect edge to a grad_fn if the next node is an intermediate tensor. Else, connect to a
    /// GradAccum for leaf nodes and return that edge
    fn connect_to_node(tensor: &Tensor<T>, input_nr: usize) -> Option<Edge<T>> {
        match tensor.get_autograd_ref() {
            Some(meta) => match &meta.grad_fn {
                Some(grad_fn_ref) => {
                    let edge = Edge {
                        input_nr: input_nr,
                        grad_fn_linked: Arc::clone(grad_fn_ref),
                    };

                    return Some(edge);
                }
                // if grad_fn is none, meaning Node does not have a function that directly
                // computes its grad, then its a leaf node
                None => match &meta.grad_accum {
                    Some(grad_accum_ref) => {
                        let edge = Edge {
                            input_nr: input_nr,
                            grad_fn_linked: Arc::clone(grad_accum_ref) as Arc<dyn Backward<T>>,
                        };

                        return Some(edge);
                    }
                    None => None,
                },
            },
            None => None,
        }
    }
}
