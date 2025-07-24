use crate::core::tensor::ops::Add;
use core::panic;
use ndarray::{Array, ArrayBase, ArrayD, IxDyn, OwnedRepr, ScalarOperand};
use num_traits::Zero;
use std::ops::{self, AddAssign};

use super::super::config::CONFIG;
use super::dtypes::DTypeMarker;
use super::storage::Storage;
use crate::core::dtypes::DTypes;

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

    pub fn get_shape(self) -> Vec<usize> {
        return self.shape;
    }

    pub fn get_strides(self) -> Vec<usize> {
        return self.strides;
    }

    pub fn get_numel(self) -> usize {
        return self.numel;
    }

    pub fn get_version(self) -> u64 {
        return self.version;
    }

    pub fn get_raw_data(&self) -> &ArrayBase<OwnedRepr<F>, IxDyn> {
        return self.storage.get_data();
    }

    pub fn get_nbytes(self) -> usize {
        return self.storage.get_nbytes();
    }

    pub fn get_type(self) -> DTypes {
        return self.storage.get_dtype();
    }
}

impl<TensorType, ScalarType> ops::Add<ScalarType> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Add<ScalarType, Output = TensorType>,
    ScalarType: AddAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn add(self, _rhs: ScalarType) -> Tensor<TensorType> {
        let data = self.get_raw_data();
        let new_raw_data = data + _rhs;

        let tensor = Tensor::from_raw_array(new_raw_data);

        return tensor;
    }
}

impl<'a, 'b, TensorType> Add<&'b Tensor<TensorType>> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Add<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'b Tensor<TensorType>) -> Self::Output {
        let new_raw_data = self.get_raw_data() + rhs.get_raw_data();
        Tensor::from_raw_array(new_raw_data)
    }
}

impl<TensorType> Add<Tensor<TensorType>> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Add<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

mod test {
    use super::*;

    #[test]
    fn create_tensor() {
        let x = vec![1, 2, 3, 4];
        let _a = Tensor::new(x, vec![4, 1]);
    }

    #[test]
    fn add_tensor() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1]);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1]);

        let _c = a + b;
    }
}
