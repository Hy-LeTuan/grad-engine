use crate::ops::compute::mul_compute::{compute_mul_tensor_scalar, compute_mul_tensor_tensor};
use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Mul, MulAssign};

impl<TensorType, ScalarType> Mul<ScalarType> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<ScalarType, Output = TensorType>,
    ScalarType: DTypeMarker + MulAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: ScalarType) -> Self::Output {
        return compute_mul_tensor_scalar(&self, rhs);
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return compute_mul_tensor_scalar(&rhs, self);
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for f64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<f64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return compute_mul_tensor_scalar(&rhs, self);
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return compute_mul_tensor_scalar(&rhs, self);
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for i64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<i64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return compute_mul_tensor_scalar(&rhs, self);
    }
}

impl<TensorType> Mul for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Self) -> Self::Output {
        return compute_mul_tensor_tensor(&self, &rhs);
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn mul_tensor() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);

        let _c = a * b;
    }

    #[test]
    fn mul_scalar() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1], false);
        let b = 3;

        let _c = a * b;
    }
}
