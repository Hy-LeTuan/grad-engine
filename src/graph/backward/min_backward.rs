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
use crate::ops::compute::add_compute::add_compute_tensor_tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct MinBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
    indices: Option<Rc<Tensor<usize>>>,
    reduced_dim: Axis,
}

impl<T> Backward<T> for MinBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T> + Zero,
{
    fn save_grad_to_origin_tensor(&self, grad: &Rc<Tensor<T>>) {
        if let Some(origin_as_option_ref) = self.origin.as_ref() {
            if let Some(origin_as_strong_rc) = origin_as_option_ref.upgrade() {
                if let Some(origin_ref) = origin_as_strong_rc.borrow().get_autograd_ref_().as_ref()
                {
                    if origin_ref.grad_is_set() {
                        let old_grad = origin_ref.get_grad_as_tensor();
                        let new_grad = add_compute_tensor_tensor(old_grad.deref(), grad.deref());

                        origin_ref.set_grad(Rc::new(new_grad));
                    } else {
                        origin_ref.set_grad(Rc::clone(grad));
                    }
                }
            }
        } else {
            panic!(
                "Dangling graph node, no origin tensor found at node: {} with id: {}",
                self.get_name(),
                self.get_id(),
            );
        }
    }

    fn apply(&self, upstream_gradient: Rc<Tensor<T>>, retain_graph: bool) {
        if retain_graph {
            self.save_grad_to_origin_tensor(&upstream_gradient);
        }

        for edge in self.get_edge_list().iter() {
            let next_grad = self.calculate_gradient_for_next_node(&upstream_gradient, Some(&edge));

            let next_node = edge.get_next_grad_fn();
            next_node.borrow().apply(next_grad, retain_graph);
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
                    "Error, trying to calculate gradient of a min function without any indices set"
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

impl<T> MinBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = MinBackward {
            input_refs: vec![],
            name: BackwardType::MinBackward,
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
    use crate::utils::testing_utils::total_test_for_backward_operation;

    #[test]
    fn min_backward_operation() {
        let x1 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();

        let z = x1.min(Axis(0));

        total_test_for_backward_operation(
            vec![&x1],
            vec![Tensor::new(vec![1, 1, 0, 0, 0, 0, 0, 0], vec![4, 2], false).as_float_32()],
            &z,
            "MinBackward",
            Tensor::new(vec![1, 2], vec![2, 1], false).as_float_32(),
        );
    }
}
