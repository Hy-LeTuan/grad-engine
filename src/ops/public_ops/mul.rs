use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Deref, Mul, MulAssign};

impl<'a, TensorType, ScalarType> Mul<ScalarType> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<ScalarType, Output = TensorType>,
    ScalarType: MulAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: ScalarType) -> Self::Output {
        let raw_array = self.get_raw_data();
        let new_array = raw_array.deref() * rhs;

        let tensor = Tensor::from_raw_array(new_array, false);

        return tensor;
    }
}

impl<TensorType, ScalarType> Mul<ScalarType> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<ScalarType, Output = TensorType>,
    ScalarType: MulAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: ScalarType) -> Self::Output {
        return &self * rhs;
    }
}

impl<'a, TensorType> Mul<&'a Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data.deref() * self;

        let tensor = Tensor::from_raw_array(new_data, false);

        return tensor;
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self * &rhs;
    }
}

impl<'a, TensorType> Mul<&'a Tensor<TensorType>> for f64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<f64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data.deref() * self;

        let tensor = Tensor::from_raw_array(new_data, false);

        return tensor;
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for f64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<f64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self * &rhs;
    }
}

impl<'a, TensorType> Mul<&'a Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data.deref() * self;

        let tensor = Tensor::from_raw_array(new_data, false);

        return tensor;
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self * &rhs;
    }
}

impl<'a, TensorType> Mul<&'a Tensor<TensorType>> for i64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<i64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data.deref() * self;

        let tensor = Tensor::from_raw_array(new_data, false);

        return tensor;
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for i64
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<i64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self * &rhs;
    }
}

// MUL FOR TENSOR AND TENSOR

impl<'a, 'b, TensorType> Mul<&'b Tensor<TensorType>> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'b Tensor<TensorType>) -> Self::Output {
        let left_raw_array = self.get_raw_data();
        let right_raw_array = rhs.get_raw_data();

        let raw_array = left_raw_array.deref() * right_raw_array.deref();

        let tensor = Tensor::from_raw_array(raw_array, false);

        return tensor;
    }
}

impl<TensorType> Mul for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Debug + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Self) -> Self::Output {
        return &self * &rhs;
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
