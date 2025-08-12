use ndarray::LinalgScalar;

use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::edge::Edge;
use crate::ops::compute::add_compute::add_compute_tensor_tensor;
use crate::ops::compute::matmul_compute::matmul_compute_tensor_tensor;
use crate::ops::compute::shape_compute::compute_transpose_tensorimpl;
use crate::tensor_core::tensor_impl::TensorImpl;
use crate::utils::shaping_utils::get_shape_to_transpose_last_2_dim;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct MatmulBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for MatmulBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T> + LinalgScalar,
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
        if let Some(e) = edge {
            let edge_index = e.get_edge_nr();
            if edge_index == 0 {
                let other_ref = &self.input_refs[1];
                let intended_shape =
                    get_shape_to_transpose_last_2_dim(other_ref.borrow().get_raw_shape());

                let other_transposed = compute_transpose_tensorimpl::<T, Vec<usize>>(
                    other_ref.deref(),
                    Some(intended_shape),
                );

                let tensor =
                    matmul_compute_tensor_tensor(upstream_gradient.deref(), &other_transposed);

                return Rc::new(tensor);
            } else {
                let other_ref = &self.input_refs[0];
                let intended_shape =
                    get_shape_to_transpose_last_2_dim(other_ref.borrow().get_raw_shape());

                let other_transposed = compute_transpose_tensorimpl::<T, Vec<usize>>(
                    other_ref.deref(),
                    Some(intended_shape),
                );
                let tensor =
                    matmul_compute_tensor_tensor(&other_transposed, upstream_gradient.deref());

                return Rc::new(tensor);
            }
        } else {
            panic!(
                "Error, no edge found to connect to and calculate gradient because matmul operation invovles 1 or more tensors."
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

impl<T> MatmulBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = MatmulBackward {
            name: BackwardType::MatmulBackward,
            input_refs: vec![],
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
        };

        return node;
    }
}

#[cfg(test)]
pub mod test {
    use std::ops::Deref;

    use crate::ops::public_ops::matmul_public::matmul_tensor_tensor;

    #[allow(unused)]
    use super::*;

    #[test]
    pub fn matmul_backward_creation() {
        let a = Tensor::new(
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
            vec![3, 2, 3],
            true,
        )
        .as_float_32();

        let b = Tensor::new(
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
            ],
            vec![3, 3, 4],
            true,
        )
        .as_float_32();

        let z = matmul_tensor_tensor(&a, &b);

        assert_eq!(
            z.get_shape().deref(),
            &vec![3usize, 2usize, 4usize],
            "Error: Output shape from matrix multiplication does not match"
        );

        assert_eq!(
            z.get_grad_fn().borrow().get_name(),
            String::from("MatmulBackward"),
            "MatmulBackward does not exist on tensor from mul operation"
        );
    }
}
