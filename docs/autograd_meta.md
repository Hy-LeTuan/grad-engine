# AutogradMeta

## Usage

`AutogradMeta` struct represents and contains all information needed for the tensor to do a complete backpropagation.

The intuition for how`AutogradMeta` should be constructed is explained through this write-up [[Implementing the Autodiff|Implementing Autodiff]].

## Definition

```rust
#[derive(Debug)]
pub struct AutogradMeta<T>
where
    T: DTComp + Debug,
{
    pub name: String,
    pub grad: RefCell<Option<Rc<Tensor<T>>>>,
    pub requires_grad: bool,
    pub grad_fn: Option<Rc<RefCell<dyn Backward<T>>>>,
    pub grad_accum: Option<Rc<RefCell<GradAccum<T>>>>,
}
```

### Fields:

1. `name`: Represents the name of the `AutogradMeta`, i.e through which operation is it constructed
2. `grad`: Represents the gradient of the Tensor
3. `requires_grad`: Boolean whether the current tensor needs its gradient tracked
4. `grad_fn`: A reference counting pointer to a `dyn Backward<T>` in the computation graph, which is a **backward function** depending on how the tensor was created. The `Backward<T>` trait is defined in [[backward]]
5. `grad_accum`: A reference counting pointer to a `GradAccum` to accumulate gradient for leaf tensors. `GradAccum` node is defined in [[grad_accum]]

### Traits

1. `Debug`: Required to easily visualize the data and the tensor for ease of debugging
2. `DTComp`: A trait that is implemented for any types that is allowed to store in a tensor. The types can be found in [[dtypes|Dtypes]]