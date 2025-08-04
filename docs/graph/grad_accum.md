# Grad Accum

## Usage

`GradAccum`is a node in the computation graph that is responsible for aggregating the gradients of a leaf tensor from all graph paths.

## Definition

```rust
#[derive(Debug)]
pub struct GradAccum<T>
where
    T: DTComp + Debug,
{
    name: BackwardType,
    id: usize,
    edge_list: Vec<Edge<T>>,
    origin: Option<Weak<RefCell<TensorImpl<T>>>>,
}
```

### Fields

1. `name`: The name of the node, defined in [[backward_type]]
2. `id`: The id of the node
3. `edge_list`: The list of edges that this node connects to. The node has to calculate the gradient for all edges it connects to
4. `origin`: The origin tensor that the node has to accumulate gradient for

### Traits

1. `Debug`: Required to easily visualize the data and the tensor for ease of debugging
2. `DTComp`: A trait that is implemented for any types that is allowed to store in a tensor. The types can be found in [[dtypes|Dtypes]]

## `Backward` implementation

```rust
impl<T> Backward<T> for GradAccum<T>
where
    T: Clone + DTComp + Debug + Add<Output = T>,
```

### New traits

1. `Clone`: Clone is needed for `Add` operation
2. `Add<Output = T>`: This trait is needed because `GradAccum` performs element wise addition to aggregate the gradient
