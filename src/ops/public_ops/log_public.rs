use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use crate::ops::central::log_impl::log_impl;
use crate::ops::compute::log_compute::log_compute_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn log_tensor<T>(tensor: &Tensor<T>, base: T) -> Tensor<T>
where
    T: Debug + DTComp + Float + 'static + ScalarOperand,
{
    let new_tensor = log_compute_tensor(tensor, base);

    if tensor.does_require_grad() {
        new_tensor.requires_grad_intermediate("Intermediate tensor from log with arbitrary base");
        log_impl(Some(tensor), &new_tensor, Some(base.clone()));
    }

    return new_tensor;
}
