use super::super::edge::Edge;
use super::super::node::Backward;
use super::DTypeMarker;

use super::Tensor;

use num_traits::Zero;
use std::fmt::Debug;
use std::sync::Arc;

#[derive(Debug)]
pub struct GradAccum<T>
where
    T: Zero + Clone + DTypeMarker,
{
    edge_list: Vec<Edge<T>>,
    origin: Option<Arc<Tensor<T>>>,
    variables: Vec<Arc<Tensor<T>>>,
}

impl<T> Backward<T> for GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    fn new() -> Self {}
    fn get_edge_list(&self) -> &[Edge<T>] {}
    fn save_input_refs(&self, input_refs: &[&crate::tensor_core::tensor::Tensor<T>]) {}
    fn add_to_node_list(&self) {}
    fn calculate_gradient(&self) -> Arc<Tensor<T>> {}

    /// save gradient to the origin tensor
    fn save_grad_to_origin_tensor(&self, tensor: Arc<Tensor<T>>) {
        match &self.origin {
            Some(tensor_ptr) => match tensor_ptr.as_ref().get_autograd_ref() {
                Some(autograd_meta_ref) => {
                    autograd_meta_ref.set_grad(tensor);
                }
                None => {}
            },
            None => {}
        }
    }
}
