use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::shape_compute::compute_transpose;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct TransposeBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    #[allow(unused)]
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
    axes_option: Option<Vec<usize>>,
}

impl<T> Backward<T> for TransposeBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
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
            if let Some(axes) = &self.axes_option {
                let mut reverse_axes: Vec<usize> = vec![0_usize; axes.len()];

                for (i, p) in axes.iter().enumerate() {
                    reverse_axes[*p] = i;
                }

                let result_tensor =
                    compute_transpose(upstream_gradient.deref(), Some(reverse_axes));
                return Rc::new(result_tensor);
            } else {
                let result_tensor = compute_transpose(upstream_gradient.deref(), Some(vec![1, 0]));
                return Rc::new(result_tensor);
            }
        } else {
            panic!(
                "Error, no edge found to connect to and calculate gradient because transpose is an operation requiring one or more tensor"
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

impl<T> TransposeBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = TransposeBackward {
            name: BackwardType::TransposeBackward,
            input_refs: vec![],
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
            axes_option: None,
        };

        return node;
    }

    pub fn save_axes_option(&mut self, axes_option: Option<Vec<usize>>) {
        self.axes_option = axes_option;
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;
    use crate::utils::testing_utils::total_test_for_backward_operation;

    #[test]
    fn transpose_backward_operation() {
        let x1 = Tensor::new(vec![1, 2, 3, 4, 5, 6, 7, 8], vec![4, 2], true).as_float_32();
        let z = x1.transpose(None);

        total_test_for_backward_operation(
            vec![&x1],
            vec![Tensor::new(vec![1, 1, 1, 1, 1, 1, 1, 1], vec![2, 4], false).as_float_32()],
            &z,
            "TransposeBackward",
            Tensor::new(vec![1, 3, 5, 7, 2, 4, 6, 8], vec![2, 4], false).as_float_32(),
        );
    }
}
