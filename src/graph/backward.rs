use super::edge::Edge;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

// Backward node types
pub mod backward_types;

// All backward node
pub mod add_backward;
pub mod div_backward;
pub mod grad_accum;
pub mod mul_backward;
pub mod sub_backward;

pub trait Backward<T>: Debug
where
    T: DTComp + Debug,
{
    /// Save the gradient received to the origin tensor
    fn save_grad_to_origin_tensor(&self, grad: &Rc<Tensor<T>>);

    /// Save received gradient to origin tensor; calculate gradient and traverse through the graph
    fn apply(&self, upstream_gradient: Rc<Tensor<T>>);

    /// Function for actual mathemtical calculation of next node's gradients
    fn calculate_gradient_for_next_node(
        &self,
        upstream_gradient: &Rc<Tensor<T>>,
        edge: Option<&Edge<T>>,
    ) -> Rc<Tensor<T>>;

    /// Get edge list
    fn get_edge_list(&self) -> &[Edge<T>];

    /// Add edge to list, this Node will also own the edge
    fn add_to_edge_list(&mut self, edge: Edge<T>);

    /// Loop through input list and link inputs with each tensor's TensorImpl
    fn save_input_refs(&mut self, input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>);

    fn clear_input_refs(&mut self) {
        self.save_input_refs(vec![]);
    }

    // MISC functions
    fn get_id(&self) -> usize;

    fn get_name(&self) -> String;
}
