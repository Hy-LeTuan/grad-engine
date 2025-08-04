# Ops_impl

## Usage

The `{ops}_impl.rs` files contain functions definition relating to the creation of the backpropagation graph and the function matching that has to happen to allow a correct computation graph. This function should never be exposed publicly and should only be call under the hood by the public APIs.

## Definition

```rust
pub fn add_impl<T>(
    lhs_tensor: Option<&Tensor<T>>,
    rhs_tensor: Option<&Tensor<T>>,
    result_tensor: &Tensor<T>,
) where
    T: DTComp + Clone + Debug + 'static + Add<Output = T>,
{
    if !result_tensor.does_require_grad() {
        return;
    }

    let mut node = AddBackward::new(0, vec![], result_tensor.__get_tensor_impl());

    match (lhs_tensor, rhs_tensor) {
        (Some(l), Some(r)) => {
            if l.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(l, 0));
            }

            if r.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(r, 1));
            }
        }
        (Some(l), None) => {
            if l.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(l, 0));
            }
        }
        (None, Some(r)) => {
            if r.does_require_grad() {
                node.add_to_edge_list(Edge::maybe_create_connect(r, 1));
            }
        }
        (None, None) => {
            return;
        }
    }

    let node = Rc::new(RefCell::new(node));
    result_tensor.set_grad_fn(node);
}
```

From the definition, you can see that the function handles the backward node creation for every scenario possible in an addition operation.