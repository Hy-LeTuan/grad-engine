use ndarray::ArrayD;
use ndarray::Axis;
use ndarray::Dimension;
use ndarray::IxDyn;
use num_traits::Zero;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct MaxBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    #[allow(unused)]
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
    indices: Option<Rc<Tensor<usize>>>,
    reduced_dim: Axis,
}

impl<T> Backward<T> for MaxBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T> + Zero,
{
    fn save_grad_to_origin_tensor(&self, _grad: &Rc<Tensor<T>>) {
        return;
    }

    fn apply(&self, upstream_gradient: Rc<Tensor<T>>) {
        for edge in self.get_edge_list().iter() {
            let next_grad = self.calculate_gradient_for_next_node(&upstream_gradient, Some(&edge));

            let next_node = edge.get_next_grad_fn();
            next_node.borrow().apply(next_grad);
        }
    }

    fn calculate_gradient_for_next_node(
        &self,
        upstream_gradient: &Rc<Tensor<T>>,
        edge: Option<&Edge<T>>,
    ) -> Rc<Tensor<T>> {
        if let Some(_) = edge {
            if let Some(indices) = self.indices.clone() {
                let indices_raw_data = indices.get_raw_data();

                let upstream_gradient_raw_data = upstream_gradient.get_raw_data();

                let intended_shape = self.input_refs[0].borrow().get_storage_().get_raw_shape();
                let mut grad_output = ArrayD::<T>::zeros(intended_shape);

                for ((i, &idx), up_grad) in indices_raw_data
                    .indexed_iter()
                    .zip(upstream_gradient_raw_data.indexed_iter())
                {
                    let mut full_index = i.as_array_view().to_vec();
                    full_index.insert(self.reduced_dim.index(), idx);
                    grad_output[IxDyn(&full_index)] = up_grad.1.clone();
                }

                let result_tensor = Tensor::from_raw_array(grad_output, false);
                return Rc::new(result_tensor);
            } else {
                panic!(
                    "Error, trying to calculate gradient of a max function without any indices set"
                );
            }
        } else {
            panic!(
                "Error, no edge found to connect to and calculate gradient because min and max requires a tensor to operate"
            );
        }
    }

    fn get_edge_list(&self) -> &[Edge<T>] {
        return &self.edge_list;
    }

    fn add_to_edge_list(&mut self, edge: Edge<T>) {
        self.edge_list.push(edge);
    }

    fn save_input_refs(&mut self, input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>) {
        self.input_refs.extend(input_refs);
    }

    fn get_id(&self) -> usize {
        return self.id;
    }

    fn get_name(&self) -> String {
        return self.name.to_string();
    }
}

impl<T> MaxBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = MaxBackward {
            input_refs: vec![],
            name: BackwardType::MaxBackward,
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
            indices: None,
            reduced_dim: Axis(0),
        };

        return node;
    }

    pub fn save_indices(&mut self, indices: Rc<Tensor<usize>>) {
        self.indices = Some(indices);
    }

    pub fn save_reduced_dim(&mut self, reduced_dim: Axis) {
        self.reduced_dim = reduced_dim;
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn max_backward_operation() {
        let x = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();

        let z = x.max(Axis(0));

        if z.does_require_grad() {
            assert_eq!(
                z.get_grad_fn().borrow().get_name(),
                String::from("MaxBackward"),
                "MaxBackward does not exist on tensor max operation"
            );
        }
    }
}
