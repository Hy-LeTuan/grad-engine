use crate::{
    ops::public_ops::broadcast_public::broadcast_tensor,
    tensor_core::{dtypes::DTComp, tensor::Tensor},
};
use std::{fmt::Debug, ops::Add};

impl<T> Tensor<T>
where
    T: Debug + DTComp + Clone + Add<Output = T> + 'static,
{
    pub fn broadcast(&self, shape: Vec<usize>) -> Self {
        return broadcast_tensor(self, shape);
    }
}
