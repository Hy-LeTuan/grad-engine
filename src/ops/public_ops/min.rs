use ndarray::Axis;
use num_traits::{Bounded, Float, Zero};
use std::fmt::Debug;

use crate::ops::public_ops::min_public::{argmin_tensor, min_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

impl<T> Tensor<T>
where
    T: Debug + DTComp + Clone + PartialOrd + Bounded + Zero + 'static + Float,
{
    pub fn min(&self, dim: Axis) -> Self {
        return min_tensor(self, dim);
    }

    pub fn argmin(&self, dim: Axis) -> Tensor<usize> {
        return argmin_tensor(self, dim);
    }
}
