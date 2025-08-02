use crate::ops::public_ops::public_mul::{mul_tensor_scalar, mul_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Mul, MulAssign};

impl<'tl, TensorType, ScalarType> Mul<ScalarType> for &'tl Tensor<TensorType>
where
    TensorType: DTypeMarker
        + Zero
        + Clone
        + Debug
        + Mul<ScalarType, Output = TensorType>
        + Mul<Output = TensorType>,
    ScalarType: DTypeMarker + MulAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: ScalarType) -> Self::Output {
        return mul_tensor_scalar(self, rhs);
    }
}

impl<'tl, TensorType> Mul<&'tl Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker
        + Zero
        + Clone
        + Debug
        + Mul<f32, Output = TensorType>
        + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return mul_tensor_scalar(rhs, self);
    }
}

impl<'tl, TensorType> Mul<&'tl Tensor<TensorType>> for f64
where
    TensorType: DTypeMarker
        + Zero
        + Clone
        + Debug
        + Mul<f64, Output = TensorType>
        + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return mul_tensor_scalar(rhs, self);
    }
}

impl<'tl, TensorType> Mul<&'tl Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker
        + Zero
        + Clone
        + Debug
        + Mul<i32, Output = TensorType>
        + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return mul_tensor_scalar(&rhs, self);
    }
}

impl<'tl, TensorType> Mul<&'tl Tensor<TensorType>> for i64
where
    TensorType: DTypeMarker
        + Zero
        + Clone
        + Debug
        + Mul<i64, Output = TensorType>
        + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return mul_tensor_scalar(&rhs, self);
    }
}

impl<'tl_a, 'tl_b, TensorType> Mul<&'tl_b Tensor<TensorType>> for &'tl_a Tensor<TensorType>
where
    TensorType:
        DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType> + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'tl_b Tensor<TensorType>) -> Self::Output {
        return mul_tensor_tensor(self, rhs);
    }
}

impl<'tl, TensorType> Mul<&'tl Tensor<TensorType>> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'tl Tensor<TensorType>) -> Self::Output {
        return mul_tensor_tensor(&self, rhs);
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn mul_tensor() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);

        let _c = &a * &b;
    }

    #[test]
    fn mul_scalar() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = 3;

        let _c = &a * b;
    }
}
