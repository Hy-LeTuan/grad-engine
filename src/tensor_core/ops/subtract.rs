use super::super::dtypes::DTypeMarker;
use super::super::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::ops::{Sub, SubAssign};

impl<'a, TensorType, ScalarType> Sub<ScalarType> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Sub<ScalarType, Output = TensorType>,
    ScalarType: SubAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: ScalarType) -> Self::Output {
        let raw_data = self.get_raw_data();
        let new_raw_data = raw_data - rhs;

        let tensor = Tensor::from_raw_array(new_raw_data);

        return tensor;
    }
}

impl<TensorType, ScalarType> Sub<ScalarType> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Sub<ScalarType, Output = TensorType>,
    ScalarType: SubAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: ScalarType) -> Self::Output {
        return &self - rhs;
    }
}

impl<'a, 'b, TensorType> Sub<&'b Tensor<TensorType>> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Sub<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: &'b Tensor<TensorType>) -> Self::Output {
        let raw_data_left = self.get_raw_data();
        let raw_data_right = rhs.get_raw_data();

        let new_raw_data = raw_data_left - raw_data_right;

        let tensor = Tensor::from_raw_array(new_raw_data);

        return tensor;
    }
}

impl<TensorType> Sub for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Sub<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn sub(self, rhs: Self) -> Self::Output {
        return &self - &rhs;
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn subtract_tensor() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1]);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1]);

        let _c = a - b;
    }

    #[test]
    fn subtract_scalar() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1]).as_float_32();
        let b = 4.0;
        let _c = a - b;
    }
}
