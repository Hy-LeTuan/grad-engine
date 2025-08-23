use std::fmt::Debug;
use std::ops::Add;

use ndarray::Axis;

use crate::ops::central::concat_impl::concat_impl;
use crate::ops::compute::stack_concat_compute::concat_compute;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn concat_tensor<T>(tensor_list: &[&Tensor<T>], dim: Axis) -> Tensor<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    let result_tensor = concat_compute(tensor_list, dim);

    concat_impl(tensor_list, &result_tensor, Axis(dim.index()));

    return result_tensor;
}
