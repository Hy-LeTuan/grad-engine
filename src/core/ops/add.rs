use super::super::dtypes::DTypeMarker;
use super::super::tensor::Tensor;

use ndarray::ScalarOperand;
use num_traits::Zero;
use std::ops::{Add, AddAssign};

// OPERATIONS FOR ADDING SCALAR TO TENSOR

impl<'a, TensorType, ScalarType> Add<ScalarType> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Add<ScalarType, Output = TensorType>,
    ScalarType: AddAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn add(self, _rhs: ScalarType) -> Tensor<TensorType> {
        let data = self.get_raw_data();
        let new_raw_data = data + _rhs;

        let tensor = Tensor::from_raw_array(new_raw_data);

        return tensor;
    }
}

impl<TensorType, ScalarType> Add<ScalarType> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Add<ScalarType, Output = TensorType>,
    ScalarType: AddAssign + ScalarOperand,
{
    type Output = Tensor<TensorType>;

    fn add(self, _rhs: ScalarType) -> Tensor<TensorType> {
        return &self + _rhs;
    }
}

impl<'a, TensorType> Add<&'a Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Add<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data + self;

        let tensor = Tensor::from_raw_array(new_data);

        return tensor;
    }
}

impl<TensorType> Add<Tensor<TensorType>> for f32
where
    TensorType: DTypeMarker + Zero + Clone + Add<f32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self + &rhs;
    }
}

impl<'a, TensorType> Add<&'a Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Add<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'a Tensor<TensorType>) -> Self::Output {
        let raw_data = rhs.get_raw_data();
        let new_data = raw_data + self;

        let tensor = Tensor::from_raw_array(new_data);

        return tensor;
    }
}

impl<TensorType> Add<Tensor<TensorType>> for i32
where
    TensorType: DTypeMarker + Zero + Clone + Add<i32, Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Tensor<TensorType>) -> Self::Output {
        return self + &rhs;
    }
}

// OPERATIONS FOR ADDING TENSOR TO TENSOR
impl<'a, 'b, TensorType> Add<&'b Tensor<TensorType>> for &'a Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Add<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: &'b Tensor<TensorType>) -> Self::Output {
        let new_raw_data = self.get_raw_data() + rhs.get_raw_data();
        Tensor::from_raw_array(new_raw_data)
    }
}

impl<TensorType> Add<Tensor<TensorType>> for Tensor<TensorType>
where
    TensorType: DTypeMarker + Zero + Clone + Add<Output = TensorType>,
{
    type Output = Tensor<TensorType>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

mod test {
    #[allow(unused)]
    use super::*;

    #[test]
    fn add_tensor() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1]);
        let b = Tensor::new(vec![5, 6, 7, 8], vec![4, 1]);

        let _c = a + b;
    }

    #[test]
    fn add_scalar() {
        let a = Tensor::new(vec![1, 2, 3, 4], vec![4, 1]);
        let b = 3;

        let _c = a + b;
    }
}
