The file implements the outline for the `Bacward` trait which all nodes of the graph has to implement. It outlines functions that requires the trait to comply through naming convention and function signatures.

```rust
pub trait Backward<T>: std::fmt::Debug
where
    T: DTypeMarker + Zero + Clone,
{
    fn calculate_gradient(&self, others: Vec<Arc<Tensor<T>>>) -> Vec<Arc<Tensor<T>>>;

    fn get_edge_list(&self) -> &[Edge<T>];

    fn add_to_node_list(&self);

    fn save_input_refs(&self, input_refs: &[&Tensor<T>]);

    fn save_grad_to_origin_tensor(&self, tensor: Arc<Tensor<T>>);
}
```