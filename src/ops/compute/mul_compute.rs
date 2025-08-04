use ndarray::ScalarOperand;
use num_traits::Signed;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Deref, Mul};

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

pub fn mul_compute_tensor_tensor<T>(lhs_scalar: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Mul<Output = T>,
{
    let x_raw = lhs_scalar.get_raw_data();
    let y_raw = rhs_tensor.get_raw_data();

    let new_raw = x_raw.deref() * y_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn mul_compute_tensor_tensorimpl<T>(
    tensor_impl: &RefCell<TensorImpl<T>>,
    tensor: &Tensor<T>,
) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Mul<Output = T>,
{
    let binding = tensor_impl.borrow();
    let x_raw = binding.get_raw_data_();
    let y_raw = tensor.get_raw_data();

    let new_raw = x_raw * y_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn mul_compute_tensorimpl_scalar<T, S>(
    tensor_impl: &RefCell<TensorImpl<T>>,
    scalar: S,
) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Mul<S, Output = T>,
    S: ScalarOperand,
{
    let binding = tensor_impl.borrow();
    let x_raw = binding.get_raw_data_();

    let new_raw = x_raw * scalar;
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn mul_compute_tensor_scalar<T, S>(tensor: &Tensor<T>, scalar: S) -> Tensor<T>
where
    S: ScalarOperand,
    T: DTComp + Clone + Debug + Mul<S, Output = T>,
{
    let x_raw = tensor.get_raw_data();

    let new_raw = x_raw.deref() * scalar;
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn mul_compute_reverse_tensor<TensorType>(tensor: &Tensor<TensorType>) -> Tensor<TensorType>
where
    TensorType: DTComp + Clone + Debug + Signed,
{
    let raw_array = tensor.get_raw_data();
    let new_array = raw_array.map(|x| -x.clone());

    let tensor = Tensor::from_raw_array(new_array, tensor.does_require_grad());

    return tensor;
}
