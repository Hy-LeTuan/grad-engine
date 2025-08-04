# Tensor

## Usage

This is the main class of the engine. The Tensor struct provides public API to all important functionalities of the engine, including tensor creation, tensor generation with specific attributes, type conversion... Any operations that involves data manipulation however, will be handled by `TensorImpl`

## Definition

```rust
pub struct Tensor<T>
where
    T: DTComp + Debug,
{
    pub(crate) tensor_impl: Rc<RefCell<TensorImpl<T>>>,
}
```

### Fields

1. `tensor_impl`: The tensor class is an owned, stack-based wrapper that contains a reference counted pointer to a `RefCell` that contains a heap-based [[tensor_impl|TensorImpl]] that stores the actual data and relevant information. `RefCell` is needed here because some operations from tensor requires borrowing a mutable reference of `TensorImpl` at runtime without actually having access to a mutable reference of the tensor.

### Traits

1. `Debug`: Required to easily visualize the data and the tensor for ease of debugging
2. `DTComp`: A trait that is implemented for any types that is allowed to store in a tensor. The types can be found in [[dtypes|Dtypes]]