use std::fmt::Debug;

use ndarray::ScalarOperand;
use num_traits::Float;

use crate::ops::public_ops::log_public::log_tensor;
use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};

impl<T> Tensor<T>
where
    T: DTComp + Debug + Float + 'static + ScalarOperand,
{
    pub fn log(&self, base: T) -> Self {
        return log_tensor(self, base);
    }
}
