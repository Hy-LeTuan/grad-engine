use core::panic;
use ndarray::{Array, ArrayD, IxDyn, IxDynImpl, ShapeBuilder};
use num_traits::Zero;

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

        let data = Array::from_shape_vec(shape, x);
        let data = match data {
            Ok(x) => x,
            Err(e) => {
                panic!("Tensor creation error, shape mismatched: {}", e.to_string());
            }
        };

        let storage = Storage::new(data, nbytes, type_signature);

        let tensor = Tensor {
            storage: storage,
            shape: vec![1, 2],
            strides: vec![1, numel],
            numel: numel,
            version: CONFIG.version,
        };

        return tensor;
    }
}

impl Tensor<f32> {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let dyn_shape = IxDyn(&shape);
        let data = ArrayD::<f32>::zeros(dyn_shape);
        let numel = data.len();
        let type_signature = DTypes::Float32;

        let nbytes = std::mem::size_of::<f32>() * (numel as usize);
        let storage = Storage::new(data, nbytes, type_signature);

        let tensor = Tensor {
            storage: storage,
            shape: vec![1, 2],
            strides: vec![1, numel],
            numel: numel,
            version: CONFIG.version,
        };

        return tensor;
    }
}

mod test {
    use super::*;

    #[test]
    fn create_tensor() {
        let x = vec![1, 2, 3, 4];
        let a = Tensor::new(x, vec![4, 1]);
    }
}
