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
    fn calculate_gradient(&self, others: Vec<Arc<Tensor<T>>>) -> Vec<Arc<Tensor<T>>>;

    fn traverse(&self);

    fn get_edge_list(&self) -> &[Edge<T>];

    fn add_to_node_list(&self);

    fn save_input_refs(&self, input_refs: &[&Tensor<T>]);

    fn save_grad_to_origin_tensor(&self, tensor: Arc<Tensor<T>>);
}
