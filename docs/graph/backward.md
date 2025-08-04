# Backward
## Usage

The `Backward<T>` trait is a mandatory trait on all backwards operations of the computation graph. This trait defines methods that all operations should have, regardless of what computation they do. It also enforces the same traits to be shared with all graph nodes, ensuring a better developer experience.

## Definition

```rust
pub trait Backward<T>: Debug
where
    T: DTComp + Debug,
{
    fn save_grad_to_origin_tensor(&self, grad: &Rc<Tensor<T>>);
	
    fn apply(&self, upstream_gradient: Rc<Tensor<T>>);
	
    fn calculate_gradient_for_next_node(
        &self,
        upstream_gradient: &Rc<Tensor<T>>,
        edge: Option<&Edge<T>>,
    ) -> Rc<Tensor<T>>;
	
    fn get_edge_list(&self) -> &[Edge<T>];
	
    fn add_to_edge_list(&mut self, edge: Edge<T>);
	
    fn save_input_refs(&mut self, input_refs: Vec<Rc<RefCell<TensorImpl<T>>>>);

    fn clear_input_refs(&mut self) {
        self.save_input_refs(vec![]);
    }

    fn get_id(&self) -> usize;
	
    fn get_name(&self) -> String;
}
```

### Methods

1. `save_grad_to_origin_tensor`: This method allows a node to save its upstream gradient to the tensor resulted from the forward operation
2. `apply`: This is the main method of any backwards node. This method will call other methods, such as `save_grad_to_origin_tensor` and `calculate_gradient_for_next_node` which will save the gradient and compute gradient for connected nodes
3. `calculate_gradient_for_next_node`: This method calculates the gradient of the nodes connected to the current node. The calculation is based on the **transposed JVP** form, which is operation dependent
4. `add_to_edge_list`: This method add an `Edge` to the edge list. The definition of `Edge` can be found in [[edge]]
5. `save_input_refs`: This method save the inputs that is used in the forward computation, since some backward operations need the initial inputs to calculate gradient
6. `clear_input_refs`: This method clears all the `input_refs` of a node
7. `get_id`: This method gets the id of the node
8. `get_name`: This method gets the name of the node

### Traits

1. `Debug`: Required to easily visualize the data and the tensor for ease of debugging
2. `DTComp`: A trait that is implemented for any types that is allowed to store in a tensor. The types can be found in [[dtypes|Dtypes]]