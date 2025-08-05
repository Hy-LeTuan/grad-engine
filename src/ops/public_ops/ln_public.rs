use num_traits::Float;
use std::fmt::Debug;

use crate::ops::central::ln_impl::ln_impl;
use crate::ops::compute::log_compute::ln_compute_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn ln_tensor<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: Debug + DTComp + Float + 'static,
{
    let result_tensor = ln_compute_tensor(tensor);

    if tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from natural log");
        ln_impl(Some(tensor), &result_tensor);
    }

    return result_tensor;
}
