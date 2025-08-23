use std::{fmt::Debug, ops::Add};

use ndarray::Axis;

use crate::{
    ops::public_ops::stack_public::stack_tensor,
    tensor_core::{dtypes::DTComp, tensor::Tensor},
};

pub fn stack<T>(tensor_list: &[&Tensor<T>], dim: Axis) -> Tensor<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    return stack_tensor(tensor_list, dim);
}
