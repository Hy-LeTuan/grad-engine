use super::super::edge::Edge;
use super::super::node::Backward;
use super::DTypeMarker;

use super::super::super::tensor_core::autograd_meta::AutogradMeta;
use super::Tensor;

use num_traits::Zero;
use std::fmt::Debug;
use std::sync::{Arc, Weak};

#[derive(Debug)]
pub struct GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<AutogradMeta<T>>>,
    inputs: Vec<Arc<Tensor<T>>>,
}

impl<T> Backward<T> for GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    fn get_edge_list(&self) -> &[Edge<T>] {
        return &self.edge_list;
    }

    fn save_input_refs(&self, _input_refs: &[&Tensor<T>]) {}

    fn add_to_node_list(&self) {}

    fn calculate_gradient(&self, _others: Vec<Arc<Tensor<T>>>) -> Vec<Arc<Tensor<T>>> {
        return vec![];
    }

    fn traverse(&self) {}

    /// save gradient to the origin tensor
    fn save_grad_to_origin_tensor(&self, _grad: Arc<Tensor<T>>) {}
}

impl<T> GradAccum<T>
where
    T: Zero + Clone + DTypeMarker + Debug,
{
    pub fn new(edge_list: Vec<Edge<T>>, inputs: Vec<Arc<Tensor<T>>>) -> Self {
        let grad_accum = GradAccum {
            edge_list: edge_list,
            origin: None,
            inputs: inputs,
        };

        return grad_accum;
    }

    pub fn set_owned_meta(&mut self, origin: Arc<AutogradMeta<T>>) {
        self.origin = Some(Arc::downgrade(&origin));
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn grad_accum_create() {
        let _grad_accum = GradAccum::<f32>::new(vec![], vec![]);
    }
}
