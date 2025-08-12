use crate::{
    graph::{
        backward::Backward,
        visualize::serialize_graph_struct::{GraphJSON, NodeJSON, TensorJSON},
    },
    tensor_core::{dtypes::DTComp, tensor::Tensor, tensor_impl::TensorImpl},
};
use serde::Serialize;
use serde_json::to_string_pretty;
use std::{
    cell::RefCell,
    fs::File,
    io::{BufWriter, Write},
    rc::Rc,
};
use std::{fmt::Debug, ops::Add};

impl<T> NodeJSON<T> {
    pub fn add_to_children(&mut self, other: NodeJSON<T>) {
        self.children.push(other);
    }
}

pub fn serialize_single_node<T>(node: Rc<RefCell<dyn Backward<T>>>) -> NodeJSON<T>
where
    T: DTComp + Debug + Clone,
{
    let name = node.borrow().get_name();

    let origin_serialized = match node.borrow().get_origin() {
        Some(origin) => serialize_tensor(origin),
        None => TensorJSON {
            data: vec![],
            offset: None,
            shape: vec![],
        },
    };

    let gradient_serialized = match node.borrow().get_origin() {
        Some(origin) => {
            let grad = origin
                .borrow()
                .get_autograd_and_expect_res()
                .get_grad_as_tensor();

            serialize_tensor(grad.__clone_ptr_to_tensor_impl())
        }
        None => TensorJSON {
            data: vec![],
            offset: None,
            shape: vec![],
        },
    };

    let node_serialized = NodeJSON {
        name: name,
        origin: origin_serialized,
        gradient: gradient_serialized,
        children: vec![],
    };

    return node_serialized;
}

pub fn serialize_tensor<T>(tensorimpl: Rc<RefCell<TensorImpl<T>>>) -> TensorJSON<T>
where
    T: DTComp + Debug + Clone,
{
    let (data, offset) = tensorimpl
        .borrow()
        .get_raw_data_()
        .to_owned()
        .into_raw_vec_and_offset();
    let shape = tensorimpl.borrow().get_raw_shape().to_vec();

    let tensor_json = TensorJSON {
        data: data,
        offset: offset,
        shape: shape,
    };

    return tensor_json;
}

pub fn serialize_and_export_graph<T>(tensor: &Tensor<T>)
where
    T: DTComp + Debug + Clone + 'static + Add<Output = T> + Serialize,
{
    let autograd_ref = tensor.get_autograd_ref();
    let root = autograd_ref
        .as_ref()
        .expect("Autograd does not exist")
        .get_grad_fn();

    if let Some(root) = root.as_ref() {
        let root_json = serialize_node_recursive(Rc::clone(root));
        let graph = GraphJSON { root: root_json };

        if let Ok(graph_json_str) = to_string_pretty(&graph) {
            let file = File::create("output/graph.json")
                .expect("Error: Creating graph.json for serializing graph failed");

            let mut writer = BufWriter::new(file);
            match writer.write_all(graph_json_str.as_bytes()) {
                Ok(_) => {}
                Err(e) => {
                    println!("{}", e);
                }
            }
        }
    }
}

pub fn serialize_node_recursive<T>(node: Rc<RefCell<dyn Backward<T>>>) -> NodeJSON<T>
where
    T: DTComp + Debug + Clone + 'static + Add<Output = T>,
{
    // create the struct and build it up through recursively calling this function
    let mut node_json = serialize_single_node(Rc::clone(&node));

    for edge in node.borrow().get_edge_list().iter() {
        let next_node = edge.get_next_grad_fn();
        let next_node_json = serialize_node_recursive(next_node);

        node_json.add_to_children(next_node_json);
    }

    return node_json;
}
