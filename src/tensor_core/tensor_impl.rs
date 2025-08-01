use super::super::config::CONFIG;
use super::autograd_meta::AutogradMeta;
use super::dtypes::{DTypeMarker, DTypes};
use super::storage::Storage;
use super::tensor::Tensor;

use ndarray::Ix2;
use ndarray::{Array, ArrayBase, IxDyn, OwnedRepr};
use num_traits::Zero;
use std::fmt::Debug;

use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
pub struct TensorImpl<T>
where
    T: Zero + Clone + DTypeMarker + Debug + 'static,
{
    pub storage: Storage<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub numel: usize,
    pub version: u64,
    pub autograd_meta: Option<AutogradMeta<T>>,
}

impl<T> TensorImpl<T>
where
    T: DTypeMarker + Zero + Clone + Debug + 'static,
{
    pub fn new(x: Vec<T>, shape: Vec<usize>) -> Self {
        let numel = x.len();
        let type_signature = T::dtype();
        let nbytes = std::mem::size_of::<T>() * (numel as usize);

        let data = Array::from_shape_vec(shape.clone(), x);
        let data = match data {
            Ok(x) => x,
            Err(e) => {
                panic!("Tensor creation error, shape mismatched: {}", e.to_string());
            }
        };

        let storage = Storage::new(data, nbytes, type_signature);

        let tensor_impl = TensorImpl {
            storage: storage,
            shape: shape,
            strides: vec![1, numel],
            numel: numel,
            version: CONFIG.version,
            autograd_meta: None,
        };

        return tensor_impl;
    }

    pub fn generate_pointer_for_tensor(tensor_impl: Self) -> Rc<RefCell<Self>> {
        return Rc::new(RefCell::new(tensor_impl));
    }

    /// Turns on grad tracking for a leaf tensor. Any intermediate tensor will always be
    /// created with gradient, so this method does not apply for them.

    pub fn from_raw_array_(x: ArrayBase<OwnedRepr<T>, IxDyn>) -> Self {
        let shape = x.shape().to_vec();
        let numel = x.len();
        let type_signature = T::dtype();
        let nbytes = std::mem::size_of::<T>() * (numel as usize);

        let storage = Storage::new(x, nbytes, type_signature);

        let tensor_impl = TensorImpl {
            storage: storage,
            shape: shape,
            strides: vec![1, numel],
            numel: numel,
            version: CONFIG.version,
            autograd_meta: None,
        };

        return tensor_impl;
    }

    pub fn get_storage_(&self) -> &Storage<T> {
        return &self.storage;
    }

    pub fn get_raw_data_(&self) -> &ArrayBase<OwnedRepr<T>, IxDyn> {
        return self.get_storage_().get_data();
    }

    pub fn get_raw_data_as_ix2_(&self) -> ArrayBase<OwnedRepr<T>, Ix2> {
        return self.get_storage_().get_data_as_ix2();
    }

    pub fn get_nbytes_(&self) -> usize {
        return self.get_storage_().get_nbytes();
    }

    pub fn get_type_(&self) -> DTypes {
        return self.get_storage_().get_dtype();
    }

    pub fn get_autograd_ref_(&self) -> &Option<AutogradMeta<T>> {
        return &self.autograd_meta;
    }

    pub fn get_autograd_and_expect_res(&self) -> &AutogradMeta<T> {
        return &self
            .autograd_meta
            .as_ref()
            .expect("Straight access to AutogradMeta failed, value does not exist");
    }

    pub fn set_autograd_meta_(&mut self, autograd_meta: AutogradMeta<T>) {
        self.autograd_meta = Some(autograd_meta);
    }

    pub fn backward_(&mut self, starting_gradient: Vec<Tensor<T>>) {
        match self.get_autograd_ref_() {
            Some(autograd_meta_arc_ref) => {
                autograd_meta_arc_ref.start_backprop_chain(starting_gradient);
            }
            None => {
                panic!("Error, calling backwards on a tensor without ")
            }
        }
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn create_tensor_impl() {
        let x = vec![1, 2, 3, 4];
        let _a = TensorImpl::new(x, vec![4, 1]);
    }
}
