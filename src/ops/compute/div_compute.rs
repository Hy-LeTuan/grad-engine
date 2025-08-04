use ndarray::ScalarOperand;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::{Deref, Div};

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;
use crate::tensor_core::tensor_impl::TensorImpl;

pub fn div_compute_tensor_tensor<T>(lhs_scalar: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Div<Output = T>,
{
    let x_raw = lhs_scalar.get_raw_data();
    let y_raw = rhs_tensor.get_raw_data();

    let new_raw = x_raw.deref() / y_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn div_compute_tensor_tensorimpl<T>(
    lhs_scalar: &RefCell<TensorImpl<T>>,
    rhs_tensor: &Tensor<T>,
) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Div<Output = T>,
{
    let binding = lhs_scalar.borrow();
    let x_raw = binding.get_raw_data_();
    let y_raw = rhs_tensor.get_raw_data();

    let new_raw = x_raw / y_raw.deref();
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}

pub fn div_compute_tensor_scalar<T, S>(tensor: &Tensor<T>, scalar: S) -> Tensor<T>
where
    S: ScalarOperand,
    T: DTComp + Clone + Debug + Div<S, Output = T>,
{
    let x_raw = tensor.get_raw_data();

    let new_raw = x_raw.deref() / scalar;
    let tensor = Tensor::from_raw_array(new_raw, false);

    return tensor;
}
