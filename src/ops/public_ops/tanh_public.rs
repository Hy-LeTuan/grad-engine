use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use crate::ops::central::tanh_impl::tanh_impl;
use crate::ops::compute::hyperbolic_compute::tanh_compute_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn tanh_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: Debug + DTComp + Float + 'static + ScalarOperand,
{
    let new_tensor = tanh_compute_tensor(tensor);

    if tensor.does_require_grad() {
        new_tensor.requires_grad_intermediate("Intermediate tensor from tanh operation");

        tanh_impl(Some(tensor), &new_tensor);
    }

    return new_tensor;
}
