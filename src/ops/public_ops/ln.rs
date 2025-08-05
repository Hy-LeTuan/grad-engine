use std::fmt::Debug;

use num_traits::Float;

use crate::ops::public_ops::ln_public::ln_tensor;
use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};

impl<T> Tensor<T>
where
    T: DTComp + Debug + Float + 'static,
{
    pub fn ln(&self) -> Self {
        return ln_tensor(self);
    }
}
