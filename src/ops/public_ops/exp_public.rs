use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use crate::ops::central::exp_impl::exp_impl;
use crate::ops::compute::exp_compute::exp_compute_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn exp_tensor<T>(tensor: &Tensor<T>, base: T) -> Tensor<T>
where
    T: Debug + DTComp + Float + 'static + ScalarOperand,
{
    let new_tensor = exp_compute_tensor(tensor, base);

    if tensor.does_require_grad() {
        new_tensor.requires_grad_intermediate("Intermediate tensor expoential with arbitrary base");
        exp_impl(Some(tensor), &new_tensor, Some(base.clone()));
    }

    return new_tensor;
}
