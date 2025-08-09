use std::fmt::Debug;
use std::ops::Add;

use ndarray::Axis;

use crate::ops::public_ops::squeeze_public::squeeze_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Debug + DTComp + Clone + 'static + Add<Output = T>,
{
    pub fn squeeze(&self, reduced_dim: Axis) -> Self {
        return squeeze_tensor(self, reduced_dim);
    }
}
