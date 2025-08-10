use ndarray::ScalarOperand;
use num_traits::Float;
use std::fmt::Debug;

use crate::ops::central::exp_impl::exp_impl;
use crate::ops::compute::exp_compute::{exp_compute_tensor, exp2_compute_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn exp_tensor<T>(tensor: &Tensor<T>, base: Option<usize>) -> Tensor<T>
where
    T: Debug + DTComp + Float + 'static + ScalarOperand,
{
    let result_tensor;
    let natural: bool;

    match base {
        Some(_) => {
            result_tensor = exp2_compute_tensor(tensor);
            natural = false;
        }
        None => {
            result_tensor = exp_compute_tensor(tensor);
            natural = true;
        }
    }

    if tensor.does_require_grad() {
        result_tensor
            .requires_grad_intermediate("Intermediate tensor expoential with arbitrary base");

        exp_impl(Some(tensor), &result_tensor, natural)
    }

    return result_tensor;
}
