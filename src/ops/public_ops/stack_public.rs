use std::fmt::Debug;
use std::ops::Add;

use ndarray::Axis;

use crate::ops::central::stack_impl::stack_impl;
use crate::ops::compute::stack_cat_compute::stack_compute;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn stack_tensor<T>(tensor_list: &[&Tensor<T>], dim: Axis) -> Tensor<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    let result_tensor = stack_compute(tensor_list, dim);

    stack_impl(tensor_list, &result_tensor, Axis(dim.index()));

    return result_tensor;
}
