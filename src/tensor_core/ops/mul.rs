use super::super::dtypes::DTypeMarker;
use super::super::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::ops::{Mul, MulAssign};

impl<'a, TensorType, ScalarType> Mul<ScalarType> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Mul<ScalarType, Output = TensorType>,
    ScalarType: MulAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: ScalarType) -> Self::Output {
        let raw_array = self.get_raw_data();
        let new_array = raw_array * rhs;

        let tensor = Tensor::from_raw_array(new_array);

        return tensor;
    }
}

impl<TensorType, ScalarType> Mul<ScalarType> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Mul<ScalarType, Output = TensorType>,
    ScalarType: MulAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: ScalarType) -> Self::Output {
        return &self * rhs;
    }
}

impl<'a, TensorType> Mul<&'a Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Mul<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data * self;

        let tensor = Tensor::from_raw_array(new_data);

        return tensor;
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Mul<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self * &rhs;
    }
}

impl<'a, TensorType> Mul<&'a Tensor<TensorType>> for f64
where
    TensorType: DTypeMarker + Zero + Clone + Mul<f64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data * self;

        let tensor = Tensor::from_raw_array(new_data);

        return tensor;
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for f64
where
    TensorType: DTypeMarker + Zero + Clone + Mul<f64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self * &rhs;
    }
}

impl<'a, TensorType> Mul<&'a Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Mul<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data * self;

        let tensor = Tensor::from_raw_array(new_data);

        return tensor;
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Mul<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self * &rhs;
    }
}

impl<'a, TensorType> Mul<&'a Tensor<TensorType>> for i64
where
    TensorType: DTypeMarker + Zero + Clone + Mul<i64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data * self;

        let tensor = Tensor::from_raw_array(new_data);

        return tensor;
    }
}

impl<TensorType> Mul<Tensor<TensorType>> for i64
where
    TensorType: DTypeMarker + Zero + Clone + Mul<i64, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self * &rhs;
    }
}

// MUL FOR TENSOR AND TENSOR

impl<'a, 'b, TensorType> Mul<&'b Tensor<TensorType>> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Mul<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn mul(self, rhs: &'b Tensor<TensorType>) -> Self::Output {
        let left_raw_array = self.get_raw_data();
        let right_raw_array = rhs.get_raw_data();

        let raw_array = left_raw_array * right_raw_array;

        let tensor = Tensor::from_raw_array(raw_array);

        return tensor;
    }
}

impl<TensorType> Mul for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Mul<Output = TensorType>,
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
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1]);
        let b = Tensor::new(vec![1, 2, 3, 4], vec![4, 1]);

        let _c = a * b;
    }

    #[test]
    fn mul_scalar() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1]);
        let b = 3;

        let _c = a * b;
    }
}
