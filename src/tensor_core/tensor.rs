use crate::graph::backward::Backward;
use crate::graph::backward::grad_accum::GradAccum;

use super::autograd_meta::AutogradMeta;
use super::dtypes::{DTypeMarker, DTypes};
use super::storage::Storage;
use super::tensor_impl::TensorImpl;

use ndarray::Ix2;
use ndarray::{ArrayBase, ArrayD, IxDyn, OwnedRepr};
use num_traits::{AsPrimitive, Zero};
use std::cell::{Ref, RefCell};
use std::fmt::Debug;
use std::rc::Rc;

#[derive(Debug)]
pub struct Tensor<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
{
    tensor_impl: Rc<RefCell<TensorImpl<T>>>,
}

impl<T> Tensor<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static,
{
    pub fn __get_tensor_impl(&self) -> &Rc<RefCell<TensorImpl<T>>> {
        return &self.tensor_impl;
    }

    pub fn __clone_ptr_to_tensor_impl(&self) -> Rc<RefCell<TensorImpl<T>>> {
        return Rc::clone(self.__get_tensor_impl());
    }

    pub fn new(x: Vec<T>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let tensor_impl = TensorImpl::new(x, shape);

        let tensor = Tensor {
            tensor_impl: TensorImpl::generate_pointer_for_tensor(tensor_impl),
        };

        if requires_grad {
            let autograd_meta = AutogradMeta::<T>::new_for_leaf(
                String::from("leaf_grad_meta"),
                tensor.__clone_ptr_to_tensor_impl(),
            );

            tensor.set_autograd_meta(autograd_meta);
        }

        return tensor;
    }

    /// Turns on grad tracking for a leaf tensor. Any intermediate tensor will always be
    /// created with gradient, so this method does not apply for them.
    pub fn requires_grad(&self) {
        let autograd_meta = AutogradMeta::<T>::new_for_leaf(
            String::from("leaf_grad_meta"),
            self.__clone_ptr_to_tensor_impl(),
        );

        self.set_autograd_meta(autograd_meta);
    }

    pub fn requires_grad_intermediate(&self, name: &str) {
        let autograd_meta = AutogradMeta::<T>::new_for_intermediate(name);

        self.set_autograd_meta(autograd_meta);
    }

    pub fn from_raw_array(x: ArrayBase<OwnedRepr<T>, IxDyn>, requires_grad: bool) -> Self {
        let tensor_impl = TensorImpl::from_raw_array_(x);

        let tensor = Tensor {
            tensor_impl: TensorImpl::generate_pointer_for_tensor(tensor_impl),
        };

        if requires_grad {
            let autograd_meta = AutogradMeta::<T>::new_for_leaf(
                String::from("leaf_grad_meta"),
                tensor.__clone_ptr_to_tensor_impl(),
            );

            tensor.set_autograd_meta(autograd_meta);
        }

        return tensor;
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let dyn_shape = IxDyn(&shape);
        let data = ArrayD::<T>::zeros(dyn_shape);

        let tensor_impl =
            TensorImpl::generate_pointer_for_tensor(TensorImpl::from_raw_array_(data));

        let tensor = Tensor { tensor_impl };

        return tensor;
    }

    fn get_storage(&self) -> Ref<Storage<T>> {
        return Ref::map(self.__get_tensor_impl().borrow(), |tensor_impl| {
            &tensor_impl.storage
        });
    }

    pub fn get_shape(&self) -> Ref<Vec<usize>> {
        return Ref::map(self.__get_tensor_impl().borrow(), |tensor_impl| {
            &(tensor_impl.shape)
        });
    }

    pub fn get_strides(&self) -> Ref<Vec<usize>> {
        return Ref::map(self.__get_tensor_impl().borrow(), |tensor_impl| {
            &(tensor_impl.strides)
        });
    }

    pub fn get_numel(&self) -> usize {
        return self.__get_tensor_impl().borrow().numel;
    }

    pub fn get_version(&self) -> u64 {
        return self.__get_tensor_impl().borrow().version;
    }

    pub fn get_raw_data(&self) -> Ref<ArrayBase<OwnedRepr<T>, IxDyn>> {
        return Ref::map(self.__get_tensor_impl().borrow(), |tensor_impl| {
            tensor_impl.get_storage_().get_data()
        });
    }

    pub fn get_raw_data_as_ix2(&self) -> ArrayBase<OwnedRepr<T>, Ix2> {
        return self.get_storage().get_data_as_ix2();
    }

    pub fn get_nbytes(&self) -> usize {
        return self.get_storage().get_nbytes();
    }

    pub fn get_type(&self) -> DTypes {
        return self.get_storage().get_dtype();
    }

    pub fn get_autograd_ref(&self) -> Ref<Option<AutogradMeta<T>>> {
        return Ref::map(self.__get_tensor_impl().borrow(), |tensor_impl_| {
            &tensor_impl_.autograd_meta
        });
    }

    pub fn set_autograd_meta(&self, autograd_meta: AutogradMeta<T>) {
        self.__get_tensor_impl()
            .borrow_mut()
            .set_autograd_meta_(autograd_meta);
    }

    pub fn does_require_grad(&self) -> bool {
        return self.__get_tensor_impl().borrow().autograd_meta.is_some();
    }

    pub fn is_leaf(&self) -> bool {
        if self
            .__get_tensor_impl()
            .borrow()
            .get_autograd_ref_()
            .is_some()
        {
            return self
                .__get_tensor_impl()
                .borrow()
                .get_autograd_ref_()
                .as_ref()
                .expect("Checking for leaf tensor failed")
                .is_leaf();
        } else {
            return false;
        }
    }

    pub fn set_grad_fn(&self, node: Rc<RefCell<dyn Backward<T>>>) {
        self.__get_tensor_impl()
            .borrow_mut()
            .autograd_meta
            .as_mut()
            .expect("..")
            .set_grad_fn_to_node(node);
    }

    pub fn set_grad_accum(&self, node: Rc<RefCell<GradAccum<T>>>) {
        self.__get_tensor_impl()
            .borrow_mut()
            .autograd_meta
            .as_mut()
            .expect("..")
            .set_grad_accum_to_accum(node);
    }

    pub fn backward(&mut self, starting_gradient: Vec<Tensor<T>>) {
        self.__get_tensor_impl()
            .borrow_mut()
            .backward_(starting_gradient);
    }

    // Display functions

    pub fn display_grad(&self) {
        let borrowed_impl = self.tensor_impl.borrow();
        match &borrowed_impl.autograd_meta {
            Some(meta) => match &meta.grad.borrow().as_ref() {
                Some(grad) => {
                    println!("Grad: {:?}", grad);
                }
                None => {
                    println!("Grad has not been computed or is None.");
                }
            },
            None => {
                println!("Tensor has no autograd metadata.");
            }
        }
    }

    pub fn display_autograd_meta(&self) {
        let borrowed_impl = self.tensor_impl.borrow();
        match &borrowed_impl.autograd_meta {
            Some(meta) => {
                println!("AutogradMeta:");
                println!("  Name: {}", meta.name);
                println!("  Requires Grad: {}", meta.requires_grad);

                // Grad
                match &*meta.grad.borrow() {
                    Some(grad) => println!("  Grad: {:?}", grad),
                    None => println!("  Grad: None"),
                }

                // GradFn
                match &meta.grad_fn {
                    Some(grad_fn) => println!("  GradFn: {:?}", grad_fn),
                    None => println!("  GradFn: None"),
                }

                // GradAccum
                match &meta.grad_accum {
                    Some(grad_accum) => println!("  GradAccum: {:?}", grad_accum),
                    None => println!("  GradAccum: None"),
                }
            }
            None => {
                println!("Tensor has no autograd metadata.");
            }
        }
    }
}

impl<T> Tensor<T>
where
    T: DTypeMarker + Zero + Debug + Clone + Copy + 'static + AsPrimitive<f32>,
{
    pub fn as_float_32(&self) -> Tensor<f32> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        tensor = Tensor::from_raw_array(new_raw_array, false);

        return tensor;
    }
}

impl<T> Tensor<T>
where
    T: Debug + DTypeMarker + Zero + Clone + Copy + 'static + AsPrimitive<f64>,
{
    pub fn as_float_64(&self) -> Tensor<f64> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        tensor = Tensor::from_raw_array(new_raw_array, false);

        return tensor;
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn create_tensor() {
        let x = vec![1, 2, 3, 4];
        let _a = Tensor::new(x, vec![4, 1], false);
    }

    #[test]
    fn create_tensor_with_grad() {
        let x = vec![1, 2, 3, 4];
        let _a = Tensor::new(x, vec![4, 1], true);
    }
}
