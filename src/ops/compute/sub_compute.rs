use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Deref, Sub};

use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn compute_sub_tensor_tensor<TensorType>(
    tensor_lhs: &Tensor<TensorType>,
    tensor_rhs: &Tensor<TensorType>,
) -> Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<Output = TensorType>,
{
    let lhs_raw = tensor_lhs.get_raw_data();
    let rhs_raw = tensor_rhs.get_raw_data();

    let new_raw = lhs_raw.deref() - rhs_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn compute_sub_tensor_scalar<TensorType, ScalarType>(
    tensor: &Tensor<TensorType>,
    scalar: ScalarType,
) -> Tensor<TensorType>
where
    ScalarType: DTypeMarker + ScalarOperand + 'static,
    TensorType: DTypeMarker + Zero + Clone + Debug + Sub<ScalarType, Output = TensorType>,
{
    let x_raw = tensor.get_raw_data();

    let new_raw = x_raw.deref() - scalar;
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}
