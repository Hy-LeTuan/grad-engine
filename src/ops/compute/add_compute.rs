use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, Deref};

use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn compute_add_tensor_tensor<TensorType>(
    x: &Tensor<TensorType>,
    y: &Tensor<TensorType>,
) -> Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<TensorType>,
{
    let x_raw = x.get_raw_data();
    let y_raw = y.get_raw_data();

    let new_raw = x_raw.deref() + y_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn compute_add_tensor_scalar<TensorType, ScalarType>(
    x: &Tensor<TensorType>,
    y: ScalarType,
) -> Tensor<TensorType>
where
    ScalarType: DTypeMarker + ScalarOperand + 'static,
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<ScalarType, Output = TensorType>,
{
    let x_raw = x.get_raw_data();

    let new_raw = x_raw.deref() + y;
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}
