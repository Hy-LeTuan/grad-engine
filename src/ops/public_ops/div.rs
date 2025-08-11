use crate::ops::public_ops::div_public::{div_tensor_scalar, div_tensor_tensor};
use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Signed;
use std::fmt::Debug;
use std::ops::{Add, Div};

impl<'tl, T, S> Div<S> for &'tl Tensor<T>
where
    T: DTComp
        + Clone
        + Debug
        + Div<Output = T>
        + Div<S, Output = T>
        + Add<Output = T>
        + 'static
        + Signed,
    S: ScalarOperand + Debug + Clone,
{
    type Output = Tensor<T>;

    fn div(self, rhs: S) -> Self::Output {
        return div_tensor_scalar(self, rhs);
    }
}

impl<T, S> Div<S> for Tensor<T>
where
    T: DTComp
        + Clone
        + Debug
        + Div<Output = T>
        + Div<S, Output = T>
        + Add<Output = T>
        + 'static
        + Signed,
    S: ScalarOperand + Debug + Clone,
{
    type Output = Tensor<T>;

    fn div(self, rhs: S) -> Self::Output {
        return &self / rhs;
    }
}

impl<'tl_in, 'tl_out, T> Div<&'tl_out Tensor<T>> for &'tl_in Tensor<T>
where
    T: DTComp
        + Clone
        + Debug
        + Div<Output = T>
        + Add<Output = T>
        + 'static
        + ScalarOperand
        + Signed,
{
    type Output = Tensor<T>;

    fn div(self, rhs: &'tl_out Tensor<T>) -> Self::Output {
        return div_tensor_tensor(self, rhs);
    }
}

impl<'tl, T> Div<&'tl Tensor<T>> for Tensor<T>
where
    T: DTComp
        + Clone
        + Debug
        + Div<Output = T>
        + Add<Output = T>
        + 'static
        + ScalarOperand
        + Signed,
{
    type Output = Tensor<T>;

    fn div(self, rhs: &'tl Tensor<T>) -> Self::Output {
        return div_tensor_tensor(&self, rhs);
    }
}

impl<T> Div<Tensor<T>> for Tensor<T>
where
    T: DTComp
        + Clone
        + Debug
        + Div<Output = T>
        + Add<Output = T>
        + 'static
        + ScalarOperand
        + Signed,
{
    type Output = Tensor<T>;

    fn div(self, rhs: Tensor<T>) -> Self::Output {
        return &self / &rhs;
    }
}
