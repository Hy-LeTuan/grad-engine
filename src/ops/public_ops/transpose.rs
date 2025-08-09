use std::fmt::Debug;
use std::ops::Add;

use crate::ops::public_ops::transpose_public::transpose_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Debug + DTComp + Clone + 'static + Add<Output = T>,
{
    pub fn transpose(&self, axes_option: Option<Vec<usize>>) -> Self {
        return transpose_tensor(self, axes_option);
    }
}
