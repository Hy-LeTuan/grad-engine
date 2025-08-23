# TensorImpl

## Usage

`TensorImpl` is the struct that does the actual logical operations related to tensors. `TensorImpl` directly owns the data, gradient information and autodiff information of the tensor, and flexibly allow access to these resources through both mutable and immutable reference because it is stored as a `RefCell` in tensor.

The end user should never have to interact with this struct directly. All of their operations should be through a `Tensor`.

## Definition

```rust
#[derive(Debug)]
pub struct TensorImpl<T>
where
    T: DTComp + Debug,
{
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub numel: usize,
    pub version: u64,
    pub storage: Storage<T>,
    pub autograd_meta: Option<AutogradMeta<T>>,
}
```

### Fields

1. `shape`: This field contains the shape of the tensor, it can be retrieved through `tensor.get_shape()`
2. `strides`: The stride of the tensor, will be use for stride storage of arrays, but since the engine is built on `ndarary` for now, this feature is unused
3. `numel`: The number of elements inside a tensor
4. `version`: The API version of the tensor
5. `storage`: This field contains the struct `Storage<T>`, found in [[storage]]. `Storage<T>` allows access to the raw data underneath the tensor, and is built upon the `ndarray` crate.
6. `autograd_meta`: The `AutogradMeta` of the tensor, found in [[autograd_meta]]. This field contains all the information the tensor needs for a full back propagation chain.

### Traits

`TensorImpl` follows the same trait bounds as its wrapper `Tensor`.

1. `Debug`: Required to easily visualize the data and the tensor for ease of debugging
2. `DTComp`: A trait that is implemented for any types that is allowed to store in a tensor. The types can be found in [[dtypes|Dtypes]]