use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};
use std::{
    fmt::Debug,
    ops::{Add, Deref},
};

pub fn test_backward_node<T>(
    input_tensor: &Tensor<T>,
    output_tensor: &Tensor<T>,
    node_name: &str,
    backward_start: Tensor<T>,
    check_correct_output: Option<Tensor<T>>,
) where
    T: DTComp + Debug + Clone + PartialEq + Add<Output = T>,
{
    // Check for correct node attachment
    assert_eq!(
        output_tensor.get_grad_fn().borrow().get_name(),
        node_name,
        "Test failed: {} does not exist on result tensor. This could mean that the backward node was never attached or that it currently has a wrong name. Name retrieved on node: {}",
        node_name,
        output_tensor.get_grad_fn().borrow().get_name().as_str()
    );

    output_tensor.backward(backward_start);

    assert_eq!(
        input_tensor
            .get_autograd_ref()
            .as_ref()
            .unwrap()
            .get_grad_as_tensor()
            .get_shape()
            .to_vec(),
        input_tensor.get_shape().to_vec(),
        "Test failed: Gradient of leaf tensor does not match its shape"
    );

    if let Some(correct_output) = check_correct_output {
        assert_eq!(
            correct_output.get_raw_data().deref(),
            output_tensor.get_raw_data().deref(),
            "Test failed: Incorrect output tensor"
        )
    }
}
