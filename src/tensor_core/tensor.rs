use ndarray::Ix2;
use ndarray::{Array, ArrayBase, ArrayD, Dimension, IxDyn, OwnedRepr};
use num_traits::{AsPrimitive, Zero};
use std::fmt::Debug;

use super::super::config::CONFIG;
use super::dtypes::DTypeMarker;
use super::dtypes::DTypes;
use super::storage::Storage;

#[derive(Debug)]
pub struct Tensor<T, D>
where
    T: Zero + Clone,
    D: Dimension,
{
    storage: Storage<T, D>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    numel: usize,
    version: u64,
    // autograd_meta: Option<AutogradMeta>,
}

pub fn compute_tensor_details<T: DTypeMarker + Zero + Clone>(
    numel: usize,
) -> (usize, DTypes, usize) {
    let type_signature = T::dtype();
    let nbytes = std::mem::size_of::<T>() * (numel as usize);

    return (numel, type_signature, nbytes);
}

impl<T, D> Tensor<T, D>
where
    T: DTypeMarker + Zero + Clone,
    D: Dimension,
{
    pub fn new(x: Vec<T>, shape: Vec<usize>) -> Tensor<T, IxDyn> {
        let (numel, type_signature, nbytes) = compute_tensor_details::<T>(x.len());

        let data: Result<ArrayBase<OwnedRepr<T>, IxDyn>, ndarray::ShapeError> =
            Array::from_shape_vec(shape.clone(), x);
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

    pub fn new_from_vec2(x: Vec<T>, shape: (usize, usize)) -> Tensor<T, Ix2> {
        let (numel, type_signature, nbytes) = compute_tensor_details::<T>(x.len());

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
            shape: vec![shape.0, shape.1],
            strides: vec![1, numel],
            numel: numel,
            version: CONFIG.version,
        };

        return tensor;
    }

    pub fn from_raw_array(x: ArrayBase<OwnedRepr<T>, D>) -> Tensor<T, D> {
        let (numel, type_signature, nbytes) = compute_tensor_details::<T>(x.len());
        let shape = x.shape().to_vec();

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

    pub fn zeros(shape: Vec<usize>) -> Tensor<T, IxDyn> {
        let dyn_shape = IxDyn(&shape);
        let data = ArrayD::<T>::zeros(dyn_shape);
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

    fn get_storage(&self) -> &Storage<T, D> {
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

    pub fn get_raw_data(&self) -> &ArrayBase<OwnedRepr<T>, D> {
        return self.get_storage().get_data();
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
}

impl<F, D> Tensor<F, D>
where
    F: DTypeMarker + Zero + Clone + Copy + 'static + AsPrimitive<f32>,
    D: Dimension,
{
    pub fn as_float_32(&self) -> Tensor<f32, D> {
        let tensor;

        let old_raw_array = self.get_raw_data();
        let new_raw_array = old_raw_array.mapv(|elem| elem.as_());

        tensor = Tensor::from_raw_array(new_raw_array);

        return tensor;
    }
}

impl<F, D> Tensor<F, D>
where
    F: DTypeMarker + Zero + Clone + Copy + 'static + AsPrimitive<f64>,
    D: Dimension,
{
    pub fn as_float_64(&self) -> Tensor<f64, D> {
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
        let _a = Tensor::<_, IxDyn>::new(x, vec![4, 1]);
    }
}
