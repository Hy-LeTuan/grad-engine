use std::fmt::Debug;
use std::ops::Add;

use crate::ops::public_ops::reshape_public::reshape_tensor;
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Debug + DTComp + Clone + 'static + Add<Output = T>,
{
    pub fn reshape(&self, axes_option: Vec<usize>) -> Self {
        return reshape_tensor(self, axes_option);
    }
}
