use std::sync::Arc;

use super::super::tensor_core::dtypes::DTypeMarker;
use super::super::tensor_core::tensor::Tensor;
use super::edge::Edge;

use num_traits::Zero;
use std::fmt::Debug;

pub mod grad_accum;

pub trait Backward<T>: Debug
where
    T: DTypeMarker + Zero + Clone + Debug,
{
    /// Save received gradient to origin tensor; calculate gradient and traverse through the graph
    fn calculate_gradient(&self, others: Arc<Vec<Tensor<T>>>) -> Arc<Vec<Tensor<T>>>;

    /// Call `self.calculate_gradient` on connected Nodes through Edge(s)
    fn traverse(&self);

    /// Get edge list
    fn get_edge_list(&self) -> &[Edge<T>];

    /// Add edge to list, this Node will also own the edge
    fn add_to_edge_list(&mut self, edge: Edge<T>);

    /// Save references of input tensors used to compute the tensor this Node belongs to
    fn save_input_refs(&mut self, input_refs: &[&Tensor<T>]);

    fn save_grad_to_origin_tensor(&self, tensor: Arc<Tensor<T>>);
}
