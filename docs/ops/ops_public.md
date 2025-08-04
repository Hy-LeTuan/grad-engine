# Ops_public

## Usage

These files are named after the convention `{ops}_public.rs`, such as `add_public.rs` or `mul_public.rs`. These files contain the definition of the engine's main public API endpoints used for tensor operation and more importantly, computation graph creation and backwards propagation tracking.

## Definition

```rust
pub fn add_tensor_tensor<T>(lhs_tensor: &Tensor<T>, rhs_tensor: &Tensor<T>) -> Tensor<T>
where
    T: DTComp + Clone + Debug + Add<Output = T> + 'static,
{
    let result_tensor = add_compute::add_compute_tensor_tensor(lhs_tensor, rhs_tensor);

    if lhs_tensor.does_require_grad() || rhs_tensor.does_require_grad() {
        result_tensor.requires_grad_intermediate("Intermediate tensor from add");
    }

    add_impl(Some(lhs_tensor), Some(rhs_tensor), &result_tensor);

    return result_tensor;
}
```

As you can see, the function `add_tensor_tensor` handles both the mathematical computation and the graph creation if gradient tracking is enabled. This function consists of 2 very important function calls:
1. `add_compute_tensor_tensor()`: This calls the functions defined in [[ops_compute]], where the actual mathematical computation happens. In this case, it is the element wise addition, hence the trait bound `Add<Output = T>`.
2. `add_impl()`: This function calls functions defined in [[ops_impl]], which handles the graph creation, specifically matching the forward operation with the corresponding backwards operation, saving the inputs to the backwards node and constructing edges to the correct gradient flow.