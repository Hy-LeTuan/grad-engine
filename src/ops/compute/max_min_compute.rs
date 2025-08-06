use ndarray::Axis;
use num_traits::{Bounded, Signed};
use std::fmt::Debug;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn min<T>(tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Signed + PartialOrd + Ord + Bounded,
{
    let raw_array = tensor.get_raw_data();
    raw_array.map_axis(Axis(0), |view| {
        view.iter()
            .fold(T::max_value(), |acc, x| T::min(acc, x.clone()))
    });

    todo!()
}
