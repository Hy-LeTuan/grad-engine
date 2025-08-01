use crate::tensor_core::dtypes::DTypeMarker;
use crate::tensor_core::tensor::Tensor;

use ndarray::Ix2;
use ndarray::LinalgScalar;
use std::fmt::Debug;

use num_traits::Zero;

impl<T> Tensor<T>
where
    T: DTypeMarker + Zero + Clone + LinalgScalar + Debug,
{
    pub fn tensor_dot(&self, rhs: &Tensor<T>) -> Tensor<T> {
        let left_raw_array = self
            .get_raw_data_as_ix2()
            .into_dimensionality::<Ix2>()
            .unwrap();

        let right_raw_array = rhs
            .get_raw_data_as_ix2()
            .into_dimensionality::<Ix2>()
            .unwrap();

        let raw_array = left_raw_array.dot(&right_raw_array).into_dyn();

        let tensor = Tensor::from_raw_array(raw_array, false);

        return tensor;
    }
}
