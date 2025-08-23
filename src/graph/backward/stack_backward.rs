use super::DTComp;
use super::Tensor;
use ndarray::Axis;

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
pub struct StackBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
    dim: Axis,
}

impl<T> Backward<T> for StackBackward<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
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
        if let Some(edge) = edge {
            let input_nr = edge.get_edge_nr();
            let stacked_array = upstream_gradient.get_raw_data();

            let unstacked_array = stacked_array.index_axis(self.dim, input_nr).to_owned();

            let tensor = Tensor::from_raw_array(unstacked_array, false);

            return Rc::new(tensor);
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

    fn get_origin(&self) -> Option<Rc<RefCell<TensorImpl<T>>>> {
        match &self.origin {
            Some(origin_weak) => {
                let origin_rc = origin_weak.upgrade();
                return origin_rc;
            }
            None => {
                panic!("no origin found on this tensor");
            }
        }
    }
}

impl<T> StackBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = StackBackward {
            input_refs: vec![],
            name: BackwardType::StackBackward,
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
            dim: Axis(0),
        };

        return node;
    }

    pub fn save_dim(&mut self, reduced_dim: Axis) {
        self.dim = reduced_dim;
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;
    use crate::{
        ops::public_ops::stack::stack, tensor,
        utils::testing_utils::total_test_for_backward_operation,
    };

    #[test]
    fn stack_backward_operation() {
        let x1 = tensor!(1.0; requires_grad=true);
        let x2 = tensor!(2.0; requires_grad=true);
        let x3 = tensor!(3.0; requires_grad=true);

        let z = stack(&vec![&x1, &x2, &x3], Axis(0));

        total_test_for_backward_operation(
            vec![&x1, &x2, &x3],
            vec![
                tensor!(1.0; requires_grad=false),
                tensor!(1.0; requires_grad=false),
                tensor!(1.0; requires_grad=false),
            ],
            &z,
            "StackBackward",
            tensor!([[1.0], [2.0], [3.0]]; requires_grad=false),
        );
    }
}
