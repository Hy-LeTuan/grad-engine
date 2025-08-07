use ndarray::{Axis, ScalarOperand};
use num_traits::NumCast;
use std::fmt::Debug;
use std::ops::{Add, Div};

use crate::ops::public_ops::mean_public::mean_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Debug
        + DTComp
        + Clone
        + 'static
        + Add<Output = T>
        + NumCast
        + Div<Output = T>
        + ScalarOperand,
{
    pub fn mean(&self, dim: Axis) -> Self {
        return mean_tensor(self, dim);
    }
}
