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


## Implementations

```rust
impl<T> Tensor<T>
where
    T: DTComp + Debug,
```

### pub fn __get_tensor_impl(&self) -> &Rc<RefCell<TensorImpl<T>>>

Get the underlying `TensorImpl` of the tensor as a reference.

```rust
use grad_engine::tensor;

let tensor = tensor!(1, 2, 3; requires_grad=false);
let tensor_impl : &Rc<RefCell<TensorImpl<_>>> = tensor.__get_tensor_impl();
```

### pub fn __clone_ptr_to_tensor_impl(&self) -> Rc<RefCell<TensorImpl<T>>>

Get the underlying `TensorImpl` of the tensor through cloning the reference counted pointer.

```rust
use grad_engine::tensor;

let tensor = tensor!(1, 2, 3; requires_grad=false);
let tensor_impl : &Rc<RefCell<TensorImpl<_>>> = tensor.__get_tensor_impl();
```

### pub fn get_raw_data(&self) -> Ref<ArrayBase<OwnedRepr<T>, IxDyn>>

Get the raw data stored in the `Storage` field of the `TensorImpl` inside a tensor. Since `grad_engine` is built upon `ndarray`, the native data storage is still an `Array` with a dynamic dimension.

```rust
use grad_engine::tensor;
use std::ops::Deref;

let tensor = tensor!(1, 2, 3; requires_grad=false);

let raw : Ref<'_, ArrayBase<OwnedRepr<i32>, Dim<IxDynImpl>>> = tensor.get_raw_data();

let raw_deref : &ArrayBase<OwnedRepr<i32>, Dim<IxDynImpl>> = raw.deref();
```

### pub fn new(x: Vec<T>, shape: Vec<usize>, requires_grad: bool) -> Self 

The *official* method to create a Tensor with any shape you want. This was the main way of creating tensors without the `tensor!` macro or if you want to create any tensors with a dimensionality > 6.

```rust
use grad_engine::tensor_core::tensor::Tensor;
let tensor : Tensor<i32> = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3], false);
println!("shape: {:?}", tensor.get_shape()) // shape: [2, 3];
```


### pub fn get_storage(&self) -> Ref<Storage<T>> 

Get the reference to the storage field of a Tensor.

```rust
use grad_engine::tensor;

let tensor = tensor!(1, 2, 3; requires_grad=false);
let storage = tensor.get_storage();
```

### pub fn get_shape(&self) -> Ref<Vec<usize>> {

Get the shape of a Tensor as a vector reference.

```rust
use grad_engine::tensor;

let tensor = tensor!(1, 2, 3; requires_grad=false);
let shape = tensor.get_shape();
println!("shape: {:?}", shape) // shape: [2, 3];
```

### pub fn get_strides(&self) -> Ref<Vec<usize>> {

Get the stride of the underlying data representation. This function is only for future migration from `ndarray` to native data representation.

### pub fn get_numel(&self) -> usize

Get the number of elements stored in the data representation. This function is only for future migration from `ndarray` to native data representation.

### pub fn get_version(&self) -> u64 

Get the tensor version iteration of the current tensor. This function is only for future migration from `ndarray` to native data representation.

### pub fn get_nbytes(&self) -> usize

Get the number of bytes of the underlying data. This function is only for future migration from `ndarray` to native data representation.

### pub fn get_type(&self) -> DTypes

Get the type of the current tensor. The type is represented through the `DTypes` struct.

```rust
use grad_engine::tensor;

let tensor = tensor!(1, 2, 3; requires_grad=false);
let dtype : DTypes = tensor.get_type();
println!("Dtype: {dtype}") // Dtype: int32;
```

### pub fn get_autograd_ref(&self) -> Ref<Option<AutogradMeta<T>>> 

Get the `autograd_meta` reference stored in the underlying `TensorImpl` of a tensor. The `autograd_meta` is an instance of the `AutogradMeta` struct. Returns an `Option` because any tensors that doesn't have gradient tracking enabled will not have an `autograd_meta`.

```rust
use grad_engine::tensor;

let tensor = tensor!(1, 2, 3; requires_grad=true);
let at : &AutogradMeta<i32> = tensor.get_autograd_ref().as_ref().unwrap();
```

### pub fn does_require_grad(&self) -> bool 

Returns `true` if the tensor has gradient tracking enable and false otherwise.

```rust
use grad_engine::tensor;

let t1 = tensor!(1, 2, 3; requires_grad=true);
println!("does require_grad: {}", t1.does_require_grad()); // true

let t2 = tensor!(1, 2, 3; requires_grad=false);
println!("does require_grad: {}", t2.does_require_grad()); // false
```

### pub fn display_grad(&self)

*This method is a part of the visualization process*

Calling this method on a tensor that has gradient tracking enabled will print the gradient of the tensor in an easy to read format. This method **panics** if the tensor does not have gradient tracking enabled. Check out the `examples` to see how this method would be called.

### pub fn display_autograd_meta(&self)

*This method is a part of the visualization process*

Calling this method on a tensor that has gradient tracking enabled will print the full `autograd_meta` information stored on the tensor. This includes all computation nodes the tensor is connected to. This method **panics** if the tensor does not have gradient tracking enabled.

### pub fn set_autograd_meta(&self, autograd_meta: AutogradMeta<T>)

Set the `autograd_meta` field of a tensor.

```rust
let tensor = tensor!(1, 2, 3; requires_grad=false);

let autograd_meta = AutogradMeta::<_>::new_for_leaf(
    String::from("leaf_grad_meta"),
    tensor.__clone_ptr_to_tensor_impl(),
);

tensor.set_autograd_meta(autograd_meta);
```

### pub fn is_leaf(&self) -> bool

Returns `true` if the tensor is a leaf tensor explicitly created by the user and `false` otherwise.

### pub fn get_grad_fn(&self) -> Rc<RefCell<dyn Backward<T>>> 

Get the computation graph backward node attached on the `autograd_meta` of a tensor.

```rust
use grad_engine::tensor;

let x1 = tensor!(1.0; requires_grad=true);
let z = stack(&vec![&x1], Axis(0));

println!("grad_fn: {:?}", z.get_grad_fn().borrow().get_name()); // StackBackward
```

### pub fn get_grad_accum(&self) -> Rc<RefCell<GradAccum<T>>> 

Get the `GradAccum` node attached on the `autograd_meta` of a tensor.

```rust
use grad_engine::tensor;

let x1 = tensor!(1.0; requires_grad=true);
println!("grad_accum: {:?}", x1.get_grad_accum().borrow().get_name()) // GradAccum;
```

### pub fn set_grad_fn(&self, node: Rc<RefCell<dyn Backward<T>>>)

Set the `grad_fn` field for a tensor to any graph node that implements the `Backward` traits.

### pub fn set_grad_accum(&self, node: Rc<RefCell<GradAccum<T>>>)

Set the `grad_accum` field for a tensor.

### pub fn requires_grad(&self)

This method creates a new `autograd_meta` attached with a `GradAccum`. This method should only be called for leaf tensors or for tensors requiring `GradAccum` to be present.

### pub fn requires_grad_intermediate(&self, name: &str)

This method creates a new `autograd_meta` with `grad_fn` and `grad_accum` set to `None`.

### pub fn from_raw_array(x: ArrayBase<OwnedRepr<T>, IxDyn>, requires_grad: bool) -> Self

Create a `Tensor` directly from an `ndarray` with a dynamic dimension.

```rust
impl<T> Tensor<T>
where
    T: DTComp + Debug + Clone + Add<Output = T>,
```

### pub fn backward(&self, starting_gradient: Tensor<T>, retain_graph: bool)

The starting point for the backpropagation process. If `retain_graph` is set to `true`, the gradients of intermediate values will also be tracked and accumulated accordingly, similar to leaf tensors.

```rust
impl<T> Tensor<T>
where
    T: DTComp + Debug + Clone,
{
```

### pub fn get_raw_data_as_ix2(&self) -> ArrayBase<OwnedRepr<T>, Ix2>

Get raw data, similar to `get_raw_data` but also casted the raw array as a 2D array. **Panics** if conversion cannot happen.

### pub fn get_raw_data_as_ix1(&self) -> ArrayBase<OwnedRepr<T>, Ix1>

Get raw data, similar to `get_raw_data` but also casted the raw array as a 1D array. **Panics** if conversion cannot happen.

## Implementation for other traits

```rust
impl<T> Display for Tensor<T>
where
    T: DTComp + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.get_shape();
        writeln!(
            f,
            "tensor(shape={:?}, dtype={}, data=",
            shape,
            self.get_type()
        )?;

        let raw_data = self.get_raw_data();
        write!(f, "  {:?}", raw_data)?;
        write!(f, ")")
    }
``
