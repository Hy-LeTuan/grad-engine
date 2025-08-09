use std::fmt::Debug;
use std::ops::Add;

use ndarray::Axis;

use crate::ops::public_ops::unsqueeze_public::unsqueeze_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Debug + DTComp + Clone + 'static + Add<Output = T>,
{
    pub fn unsqueeze(&self, reduced_dim: Axis) -> Self {
        return unsqueeze_tensor(self, reduced_dim);
    }
}
