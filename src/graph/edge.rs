use super::backward::Backward;

use super::super::tensor_core::dtypes::DTypeMarker;
use super::super::tensor_core::tensor::Tensor;

use num_traits::Zero;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

#[derive(Debug)]
pub struct Edge<T>
where
    T: DTypeMarker + Zero + Clone + Debug,
{
    pub grad_fn_linked: Rc<RefCell<dyn Backward<T>>>,
    pub input_nr: usize,
}

impl<T> Edge<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static,
{
    /// Connect edge to a grad_fn if the next node is an intermediate tensor. Else, connect to a
    /// GradAccum for leaf nodes and return that edge
    pub fn create_and_connect_to_node(tensor: &Tensor<T>, input_nr: usize) -> Option<Edge<T>> {
        match tensor.get_autograd_ref().as_ref() {
            Some(meta) => match &meta.grad_fn {
                Some(grad_fn_ref) => {
                    let edge = Edge {
                        input_nr: input_nr,
                        grad_fn_linked: Rc::clone(grad_fn_ref),
                    };

                    return Some(edge);
                }
                // if grad_fn is none, meaning Node does not have a function that directly
                // computes its grad, then its a leaf node
                None => match &meta.grad_accum {
                    Some(grad_accum_ref) => {
                        let edge = Edge {
                            input_nr: input_nr,
                            grad_fn_linked: Rc::clone(grad_accum_ref)
                                as Rc<RefCell<dyn Backward<T>>>,
                        };

                        return Some(edge);
                    }
                    None => None,
                },
            },
            None => None,
        }
    }

    pub fn get_next_grad_fn(&self) -> Rc<RefCell<dyn Backward<T>>> {
        return Rc::clone(&self.grad_fn_linked);
    }

    pub fn set_edge_nr(&mut self, new_input_nr: usize) {
        self.input_nr = new_input_nr;
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn edge_creation_and_methods() {
        let tensor = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], true);
        let _edge = Edge::create_and_connect_to_node(&tensor, 1);
    }
}
