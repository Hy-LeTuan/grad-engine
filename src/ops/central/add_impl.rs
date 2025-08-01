use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, Deref};

use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

pub fn add_impl<TensorType>(
    lhs_tensor: &Tensor<TensorType>,
    rhs_tensor: &Tensor<TensorType>,
    result_tensor: &Tensor<TensorType>,
) where
    TensorType: DTypeMarker + Zero + Clone + Debug + Add<TensorType>,
{
}
