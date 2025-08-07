use ndarray::Axis;
use std::fmt::Debug;
use std::ops::Add;

use crate::ops::public_ops::sum_public::sum_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Debug + DTComp + Clone + 'static + Add<Output = T>,
{
    pub fn sum(&self, dim: Axis) -> Self {
        return sum_tensor(self, dim);
    }
}
