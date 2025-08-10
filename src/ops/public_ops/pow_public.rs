use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use crate::ops::central::pow_impl::pow_impl;
use crate::ops::compute::pow_compute::pow_compute_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn pow_tensor<T>(tensor: &Tensor<T>, base: T) -> Tensor<T>
where
    T: Debug + DTComp + Float + 'static + ScalarOperand,
{
    let new_tensor = pow_compute_tensor(tensor, base);

    if tensor.does_require_grad() {
        new_tensor.requires_grad_intermediate("Intermediate tensor expoential with arbitrary base");
        pow_impl(Some(tensor), &new_tensor, Some(base.clone()));
    }

    return new_tensor;
}
