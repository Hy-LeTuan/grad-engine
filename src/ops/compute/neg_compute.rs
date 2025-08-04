use num_traits::Signed;
use std::fmt::Debug;

use crate::tensor_core::dtypes::DTComp;
use crate::tensor_core::tensor::Tensor;

pub fn neg_compute_tensor<TensorType>(tensor: &Tensor<TensorType>) -> Tensor<TensorType>
where
    TensorType: DTComp + Clone + Debug + Signed,
{
    let raw_array = tensor.get_raw_data();
    let new_array = raw_array.map(|x| -x.clone());

    let tensor = Tensor::from_raw_array(new_array, tensor.does_require_grad());

    return tensor;
}
