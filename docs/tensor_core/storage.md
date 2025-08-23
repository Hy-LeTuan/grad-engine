# Storage

## Usage

`Storage` struct is the struct containing the raw data for every tensor. Even though operations are defined on the tensors themselves, the data will be queried from `Storage` first. Right now, the engine is built on the `ndarray` crate, which is why storage is a wrapper around an `ArrayBase<>`, but this is subject to change in the future.

## Definition

```rust
#[derive(Debug)]
pub struct Storage<T>
where
    T: DTComp,
{
    data: ArrayBase<OwnedRepr<T>, IxDyn>,
    nbytes: usize,
    dtype: dtypes::DTypes,
}
```

### Fields

1. `data`: The `ndarray` that contains the pure data
2. `nbytes`: The number of bytes this data takes
3. `dtype`: The type of data stored in the tensor. Can generate this `Enum` on any type that implements the `DTComp` trait. You can check out the allowed types in [[dtypes]].

### Traits

1. `DTComp`: A trait that is implemented for any types that is allowed to store in a tensor. The types can be found in [[dtypes|Dtypes]]