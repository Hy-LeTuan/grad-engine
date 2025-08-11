use super::DTComp;
use super::Tensor;

use crate::graph::backward::Backward;
use crate::graph::backward::backward_types::BackwardType;
use crate::graph::backward::backward_utils::gradient_from_broadcast;
use crate::graph::edge::Edge;
use crate::tensor_core::tensor_impl::TensorImpl;

use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Add;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct AddBackward<T>
where
    T: DTComp + Clone + Debug,
{
    input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>,
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    #[allow(unused)]
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}

impl<T> Backward<T> for AddBackward<T>
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
        if let Some(edge) = edge {
            let edge_nr = edge.input_nr;

            let input_tensor;

            if edge_nr == 0 {
                input_tensor = Rc::clone(&self.input_refs[0]);
            } else {
                input_tensor = Rc::clone(&self.input_refs[1]);
            }

            return Rc::new(gradient_from_broadcast(
                upstream_gradient.deref(),
                &input_tensor.borrow().get_raw_shape(),
            ));
        } else {
            panic!("Cannot calculate gradient for add operation because of missing inputs");
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

impl<T> AddBackward<T>
where
    T: Clone + DTComp + Debug,
{
    pub fn new(id: usize, edge_list: Vec<Edge<T>>, origin: &Rc<RefCell<TensorImpl<T>>>) -> Self {
        let node = AddBackward {
            input_refs: vec![],
            name: BackwardType::AddBackward,
            id,
            edge_list,
            origin: Some(Rc::downgrade(origin)),
        };

        return node;
    }
}

#[cfg(test)]
pub mod test {
    #[allow(unused)]
    use super::*;
    use crate::utils::testing_utils::total_test_for_backward_operation;

    #[test]
    fn add_backward_operation() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true).as_float_32();
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();
        let c = Tensor::new(vec![5, 6, 7, 8], vec![4, 1], true).as_float_32();

        let d = &a + &b + &c;
        let z = &d + 3.0;

        total_test_for_backward_operation(
            vec![&a, &b, &c],
            vec![
                Tensor::new(vec![1, 1, 1, 1], vec![4, 1], false).as_float_32(),
                Tensor::new(vec![1, 1, 1, 1], vec![4, 1], false).as_float_32(),
                Tensor::new(vec![1, 1, 1, 1], vec![4, 1], false).as_float_32(),
            ],
            &z,
            "AddBackward",
            Tensor::new(vec![14, 17, 20, 23], vec![4, 1], false).as_float_32(),
        );
    }
}
