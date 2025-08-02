use ndarray::ScalarOperand;
use num_traits::{Signed, Zero};
use std::fmt::Debug;
use std::ops::{Deref, Mul};

use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn compute_mul_tensor_tensor<TensorType>(
    lhs_scalar: &Tensor<TensorType>,
    rhs_tensor: &Tensor<TensorType>,
) -> Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType>,
{
    let x_raw = lhs_scalar.get_raw_data();
    let y_raw = rhs_tensor.get_raw_data();

    let new_raw = x_raw.deref() * y_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn compute_mul_tensor_scalar<TensorType, ScalarType>(
    tensor: &Tensor<TensorType>,
    scalar: ScalarType,
) -> Tensor<TensorType>
where
    ScalarType: DTypeMarker + ScalarOperand + 'static,
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<ScalarType, Output = TensorType>,
{
    let x_raw = tensor.get_raw_data();

    let new_raw = x_raw.deref() * scalar;
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn compute_mul_reverse_tensor<TensorType>(tensor: &Tensor<TensorType>) -> Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Signed,
{
    let raw_array = tensor.get_raw_data();
    let new_array = raw_array.map(|x| -x.clone());

    let tensor = Tensor::from_raw_array(new_array, tensor.does_require_grad());

    return tensor;
}
