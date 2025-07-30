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
    inputs: Vec<Arc<Tensor<T>>>,
}

impl<T> Backward<T> for GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    fn get_edge_list(&self) -> &[Edge<T>] {
        return &self.edge_list;
    }

    fn save_input_refs(&self, input_refs: &[&Tensor<T>]) {}

    fn add_to_node_list(&self) {}

    fn calculate_gradient(&self, others: Vec<Arc<Tensor<T>>>) -> Vec<Arc<Tensor<T>>> {
        return vec![];
    }

    fn traverse(&self) {}

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

impl<T> GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    pub fn new(
        origin: Arc<Tensor<T>>,
        edge_list: Vec<Edge<T>>,
        inputs: Vec<Arc<Tensor<T>>>,
    ) -> Self {
        let grad_accum = GradAccum {
            edge_list: edge_list,
            origin: Some(origin),
            inputs: inputs,
        };

        return grad_accum;
    }
}
