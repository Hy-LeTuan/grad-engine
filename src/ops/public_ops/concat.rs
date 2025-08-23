use std::{fmt::Debug, ops::Add};

use ndarray::Axis;

use crate::{
    ops::public_ops::concat_public::concat_tensor,
    tensor_core::{dtypes::DTComp, tensor::Tensor},
};

pub fn concat<T>(tensor_list: &[&Tensor<T>], dim: Axis) -> Tensor<T>
where
    T: Clone + DTComp + Debug + 'static + Add<Output = T>,
{
    return concat_tensor(tensor_list, dim);
}
