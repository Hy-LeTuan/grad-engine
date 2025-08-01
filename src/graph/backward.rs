use super::super::tensor_core::dtypes::DTypeMarker;
use super::super::tensor_core::tensor::Tensor;
use super::edge::Edge;

use num_traits::Zero;
use std::fmt::Debug;
use std::rc::Rc;

// All backward node
pub mod add_backward;
pub mod grad_accum;
pub mod subtract_backward;

pub trait Backward<T>: Debug
where
    T: DTypeMarker + Zero + Clone + Debug,
{
    /// Save the gradient received to the origin tensor
    fn save_grad_to_origin_tensor(&self, grad: &Rc<Tensor<T>>);

    /// Save received gradient to origin tensor; calculate gradient and traverse through the graph
    fn apply(&self, upstream_gradient: Rc<Tensor<T>>);

    /// Function for actual mathemtical calculation of next node's gradients
    fn calculate_gradient_for_next_node(&self, upstream_gradient: &Rc<Tensor<T>>) -> Rc<Tensor<T>>;

    /// Get edge list
    fn get_edge_list(&self) -> &[Edge<T>];

    /// Add edge to list, this Node will also own the edge
    fn add_to_edge_list(&mut self, edge: Edge<T>);

    /// Loop through input list and link inputs with each tensor's TensorImpl
    fn save_input_refs(&mut self, input_refs: &[&Tensor<T>]);

    // MISC functions
    fn get_id(&self) -> usize;
}
