use ndarray::{Array, ArrayBase, ArrayD, IxDyn, OwnedRepr};
use num_traits::{AsPrimitive, Zero};
use std::fmt::Debug;

use super::super::config::CONFIG;
use super::dtypes::DTypeMarker;
use super::storage::Storage;
use crate::core::dtypes::DTypes;

#[derive(Debug)]
pub struct Tensor<F>
where
    F: Zero + Clone,
{
    storage: Storage<F>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    numel: usize,
    version: u64,
    // autograd_meta: Option<AutogradMeta>,
}

impl<F> Tensor<F>
where
    F: DTypeMarker + Zero + Clone,
{
    pub fn new(x: Vec<F>, shape: Vec<usize>) -> Self {
        let numel = x.len();
        let type_signature = F::dtype();
        let nbytes = std::mem::size_of::<F>() * (numel as usize);

        let data = Array::from_shape_vec(shape.clone(), x);
        let data = match data {
            Ok(x) => x,
            Err(e) => {
                panic!("Tensor creation error, shape mismatched: {}", e.to_string());
            }
        };

        let storage = Storage::new(data, nbytes, type_signature);

        let tensor = Tensor {
            storage: storage,
            shape: shape,
            strides: vec![1, numel],
            numel: numel,
            version: CONFIG.version,
        };

        return tensor;
    }

    pub fn from_raw_array(x: ArrayBase<OwnedRepr<F>, IxDyn>) -> Self {
        let shape = x.shape().to_vec();
        let numel = x.len();
        let type_signature = F::dtype();
        let nbytes = std::mem::size_of::<F>() * (numel as usize);

        let storage = Storage::new(x, nbytes, type_signature);

        let tensor = Tensor {
            storage: storage,
            shape: shape,
            strides: vec![1, numel],
            numel: numel,
            version: CONFIG.version,
        };

        return tensor;
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let dyn_shape = IxDyn(&shape);
        let data = ArrayD::<F>::zeros(dyn_shape);
        let numel = data.len();
        let type_signature = DTypes::Float32;

        let nbytes = std::mem::size_of::<f32>() * (numel as usize);
        let storage = Storage::new(data, nbytes, type_signature);

        let tensor = Tensor {
            storage: storage,
            shape: shape,
            strides: vec![1, numel],
            numel: numel,
            version: CONFIG.version,
        };

        return tensor;
    }

    fn get_storage(&self) -> &Storage<F> {
        return &self.storage;
    }

    pub fn get_shape(&self) -> &[usize] {
        return &(self.shape);
    }

    pub fn get_strides(&self) -> &[usize] {
        return &(self.strides);
    }

    pub fn get_numel(&self) -> usize {
        return self.numel;
    }

    pub fn get_version(&self) -> u64 {
        return self.version;
    }

    pub fn get_raw_data(&self) -> &ArrayBase<OwnedRepr<F>, IxDyn> {
        return self.get_storage().get_data();
    }

    pub fn get_nbytes(&self) -> usize {
        return self.get_storage().get_nbytes();
    }

    pub fn get_type(&self) -> DTypes {
        return self.get_storage().get_dtype();
    }
}

impl<F> Tensor<F>
where
    F: DTypeMarker + Zero + Clone + Copy + 'static + AsPrimitive<f32>,
{
    pub fn as_float_32(&self) -> Tensor<f32> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        tensor = Tensor::from_raw_array(new_raw_array);

        return tensor;
    }
}

impl<F> Tensor<F>
where
    F: DTypeMarker + Zero + Clone + Copy + 'static + AsPrimitive<f64>,
{
    pub fn as_float_64(&self) -> Tensor<f64> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        tensor = Tensor::from_raw_array(new_raw_array);

        return tensor;
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn create_tensor() {
        let x = vec![1, 2, 3, 4];
        let _a = Tensor::new(x, vec![4, 1]);
    }
}
