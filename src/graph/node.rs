use std::sync::Arc;

use super::super::tensor_core::dtypes::DTypeMarker;
use super::super::tensor_core::tensor::Tensor;
use super::edge::Edge;

use num_traits::Zero;

pub mod grad_accum;

pub trait Backward<T>: std::fmt::Debug
where
    T: DTypeMarker + Zero + Clone,
{
    fn calculate_gradient(&self) -> Arc<Tensor<T>>;

    fn get_edge_list(&self) -> &[Edge<T>];

    fn add_to_node_list(&self);

    fn save_input_refs(&self, input_refs: &[&Tensor<T>]);

    fn save_grad_to_origin_tensor(&self, tensor: Arc<Tensor<T>>);
}
