use ndarray::Axis;
use ndarray::ScalarOperand;
use num_traits::NumCast;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::div_compute::div_compute_tensor_scalar;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Div;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct MeanBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    #[allow(unused)]
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
    reduced_dim: Axis,
}

impl<T> Backward<T> for MeanBackward<T>
where
    T: Clone
        + DTComp
        + Debug
        + 'static
        + Add<Output = T>
        + Div<Output = T>
        + ScalarOperand
        + NumCast,
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
            let upstream_gradient_raw_data = upstream_gradient.get_raw_data().clone();
            let expanded_data = upstream_gradient_raw_data.insert_axis(self.reduced_dim);

            let shape = self.input_refs[0].borrow().get_storage_().get_raw_shape();
            let num_elem = T::from(shape[self.reduced_dim.index()])
                .expect("Error: Could not convert axis length to scalar type for mean backward");

            let broadcasted_data = expanded_data
                .broadcast(shape)
                .expect("Error: Cannot cast gradient to the correct input shape")
                .to_owned();

            let tensor = Tensor::from_raw_array(broadcasted_data, false);
            let result_tensor = div_compute_tensor_scalar(&tensor, num_elem);

            return Rc::new(result_tensor);
        } else {
            panic!(
                "Error: No edge found to connect to and calculate gradient because mean is a self operation"
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

impl<T> MeanBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = MeanBackward {
            name: BackwardType::MeanBackward,
            input_refs: vec![],
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
            reduced_dim: Axis(0),
        };

        return node;
    }

    pub fn save_reduced_dim(&mut self, dim: Axis) {
        self.reduced_dim = dim;
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn mean_backward_operation() {
        let x = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();

        let z = x.mean(Axis(0));

        if z.does_require_grad() {
            assert_eq!(
                z.get_grad_fn().borrow().get_name(),
                String::from("MeanBackward"),
                "MeanBackward does not exist on tensor mean operation"
            );
        }
    }
}
