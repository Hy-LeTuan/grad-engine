use std::fmt::Debug;

use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};

pub fn handle_requires_grad<T>(tensor: &Tensor<T>, requires_grad: Option<bool>)
where
    T: DTComp + Debug,
{
    match requires_grad {
        Some(does_require_grad) => {
            if does_require_grad {
                tensor.requires_grad();
            }
        }
        None => {
            return;
        }
    }
}
