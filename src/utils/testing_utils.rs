use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use num_traits::One;

use crate::tensor_core::{dtypes::DTComp, tensor::Tensor};
use std::{cell::Ref, fmt::Debug, ops::Add};

pub fn epsilon_test_for_tensor_similarity<T>(
    y_raw: Ref<ArrayBase<OwnedRepr<T>, IxDyn>>,
    target_raw: Ref<ArrayBase<OwnedRepr<T>, IxDyn>>,
    epsilon: f64,
) where
    T: DTComp + Debug + Clone + PartialEq + Add<Output = T> + Into<f64>,
{
    for (a, b) in y_raw.iter().zip(target_raw.iter()) {
        let a_f64: f64 = a.clone().into();
        let b_f64: f64 = b.clone().into();

        assert!(
            (a_f64 - b_f64).abs() <= epsilon,
            "Test failed: Gradient mismatch. Expected {:?}, got {:?}, diff = {:.8}",
            b,
            a,
            (a_f64 - b_f64).abs()
        );
    }
}

/// This function assumes that backward has been called and all gradient has been propagated back
/// to input tensors
pub fn test_for_correct_gradient<T>(
    input_tensors: Vec<&Tensor<T>>,
    target_gradients: Vec<Tensor<T>>,
    epsilon: f64,
) where
    T: DTComp + Debug + Clone + PartialEq + Add<Output = T> + Into<f64>,
{
    for (i, (y, target)) in input_tensors
        .iter()
        .zip(target_gradients.iter())
        .enumerate()
    {
        let y_grad_tensor = y.get_autograd_ref().as_ref().unwrap().get_grad_as_tensor();
        let y_raw = y_grad_tensor.get_raw_data();

        let target_raw = target.get_raw_data();

        // Compare element-by-element with tolerance
        for (a, b) in y_raw.iter().zip(target_raw.iter()) {
            let a_f64: f64 = a.clone().into();
            let b_f64: f64 = b.clone().into();

            assert!(
                (a_f64 - b_f64).abs() <= epsilon,
                "Test failed: Gradient mismatch at pair {}. Expected {:?}, got {:?}, diff = {:.8}",
                i,
                b,
                a,
                (a_f64 - b_f64).abs()
            );
        }
    }
}

pub fn test_node_name<T>(output_tensor: &Tensor<T>, node_name: &str)
where
    T: DTComp + Debug + Clone,
{
    assert_eq!(
        output_tensor.get_grad_fn().borrow().get_name(),
        node_name,
        "Test failed: {} does not exist on result tensor. This could mean that the backward node was never attached or that it currently has a wrong name. Name retrieved on node: {}",
        node_name,
        output_tensor.get_grad_fn().borrow().get_name().as_str()
    );
}

/// Testing function for bacward nodes. The function checks for correct grad_fn name on output,
/// correct output and correct gradient
pub fn total_test_for_backward_operation<T>(
    input_tensors: Vec<&Tensor<T>>,
    target_gradients: Vec<Tensor<T>>,
    output_tensor: &Tensor<T>,
    node_name: &str,
    target_output: Tensor<T>,
) where
    T: DTComp + Debug + Clone + PartialEq + Add<Output = T> + One + Into<f64>,
{
    let epsilon = 1e-4;
    test_node_name(output_tensor, node_name);

    // call backward and propagate gradient to input
    output_tensor.backward(Tensor::ones_like(output_tensor));

    test_for_correct_gradient(input_tensors, target_gradients, epsilon);

    epsilon_test_for_tensor_similarity(
        target_output.get_raw_data(),
        output_tensor.get_raw_data(),
        epsilon,
    );
}
