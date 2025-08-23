use crate::{
    graph::{
        backward::Backward,
        visualize::serialize_graph_struct::{GraphJSON, NodeJSON, NodeJSONAcyclic, TensorJSON},
    },
    tensor_core::{dtypes::DTComp, tensor::Tensor, tensor_impl::TensorImpl},
};
use serde::Serialize;
use serde_json::to_string_pretty;
use std::{
    cell::RefCell,
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    rc::Rc,
};
use std::{fmt::Debug, fs, ops::Add, path::Path};

impl<T> NodeJSON<T> {
    pub fn add_to_children(&mut self, other: NodeJSON<T>) {
        self.children.push(other);
    }
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

pub fn export_single_node<T>(
    node: Rc<RefCell<dyn Backward<T>>>,
    node_id: String,
    tensor_registry: &mut HashMap<*const (), String>,
    root_dir: &str,
) where
    T: DTComp + Debug + Clone + 'static + Add<Output = T> + Serialize,
{
    let name = node.borrow().get_name();

    let origin_serialized = match node.borrow().get_origin() {
        Some(origin) => {
            let ptr = Rc::as_ptr(&origin) as *const ();

            if tensor_registry.contains_key(&ptr) {
                String::clone(&tensor_registry[&ptr])
            } else {
                let origin_id = format!("t-{}", tensor_registry.len());
                tensor_registry.insert(ptr, String::clone(&origin_id));

                // export newly recorded tensor
                export_single_tensor(origin, String::clone(&origin_id), root_dir);

                origin_id
            }
        }
        None => panic!("Export Error: No origin found on node"),
    };

    let gradient_serialized = match node.borrow().get_origin() {
        Some(origin) => {
            let grad = origin
                .borrow()
                .get_autograd_and_expect_res()
                .get_grad_as_tensor()
                .__clone_ptr_to_tensor_impl();

            let ptr = Rc::as_ptr(&grad) as *const ();

            if tensor_registry.contains_key(&ptr) {
                String::clone(&tensor_registry[&ptr])
            } else {
                let grad_id = format!("g-{}", tensor_registry.len());
                tensor_registry.insert(ptr, String::clone(&grad_id));

                // export newly recorded tensor
                export_single_tensor(grad, String::clone(&grad_id), root_dir);

                grad_id
            }
        }
        None => panic!("Export Error: No gradient found on node"),
    };

    let node_serialized = NodeJSONAcyclic {
        name: name,
        origin: origin_serialized,
        gradient: gradient_serialized,
    };

    if let Ok(node_json_str) = to_string_pretty(&node_serialized) {
        let file = File::create(format!("{}/nodes/{}.json", root_dir, node_id))
            .expect("Error: Creating graph.json for serializing graph failed");
        let mut writer = BufWriter::new(file);
        match writer.write_all(node_json_str.as_bytes()) {
            Ok(_) => {}
            Err(e) => {
                println!("{}", e);
            }
        }
    }
}

pub fn export_single_tensor<T>(
    tensorimpl: Rc<RefCell<TensorImpl<T>>>,
    tensor_id: String,
    root_dir: &str,
) where
    T: DTComp + Debug + Clone + 'static + Add<Output = T> + Serialize,
{
    let serialized_tensor = serialize_tensor(tensorimpl);

    if let Ok(tensor_json_str) = to_string_pretty(&serialized_tensor) {
        let file = File::create(format!("{}/tensors/{}.json", root_dir, tensor_id))
            .expect("Error: Creating graph.json for serializing graph failed");
        let mut writer = BufWriter::new(file);
        match writer.write_all(tensor_json_str.as_bytes()) {
            Ok(_) => {}
            Err(e) => {
                println!("{}", e);
            }
        }
    }
}
pub fn populate_and_record_tensors_and_nodes<T>(
    node: Rc<RefCell<dyn Backward<T>>>,
    node_registry: &mut HashMap<*const (), String>,
    adjacency_list: &mut Vec<(String, String)>,
    tensor_registry: &mut HashMap<*const (), String>,
    root_dir: &str,
) where
    T: DTComp + Debug + Clone + 'static + Add<Output = T> + Serialize,
{
    let node_ptr = Rc::as_ptr(&node) as *const ();

    let node_name;

    if node_registry.contains_key(&node_ptr) {
        node_name = String::clone(&node_registry[&node_ptr]);
    } else {
        node_name = format!("n-{}", node_registry.len());
        node_registry.insert(node_ptr, String::clone(&node_name));

        export_single_node(
            Rc::clone(&node),
            String::clone(&node_name),
            tensor_registry,
            root_dir,
        );
    }

    for edge in node.borrow().get_edge_list().iter() {
        let next_node = edge.get_next_grad_fn();
        let next_node_ptr = Rc::as_ptr(&next_node) as *const ();
        let next_node_name;

        if node_registry.contains_key(&next_node_ptr) {
            next_node_name = String::clone(&node_registry[&next_node_ptr]);
        } else {
            next_node_name = format!("n-{}", node_registry.len());
            node_registry.insert(next_node_ptr, String::clone(&next_node_name));

            export_single_node(
                Rc::clone(&next_node),
                String::clone(&next_node_name),
                tensor_registry,
                root_dir,
            );
        }

        adjacency_list.push((String::clone(&node_name), String::clone(&next_node_name)));
        populate_and_record_tensors_and_nodes(
            next_node,
            node_registry,
            adjacency_list,
            tensor_registry,
            root_dir,
        );
    }
}

pub fn export_graph_acyclic<T>(tensor: &Tensor<T>, root: Option<String>)
where
    T: DTComp + Debug + Clone + 'static + Add<Output = T> + Serialize,
{
    let root_dir;
    match root {
        Some(root) => {
            root_dir = root;
        }
        None => {
            root_dir = String::from("output");
        }
    };

    println!("Root dir: {root_dir}");

    // create full output directory
    let output_dir_path = Path::new(root_dir.as_str());
    let tensor_dir = String::clone(&root_dir) + "/tensors";
    let node_dir = String::clone(&root_dir) + "/nodes";

    let tensor_dir_path = Path::new(&tensor_dir);
    let node_dir_path = Path::new(&node_dir);

    // check for existence, clear and create
    if output_dir_path.is_dir() {
        match fs::remove_dir_all(output_dir_path) {
            Ok(_) => {}
            Err(e) => eprintln!(
                "Export Error: Failed to clear out current output directory for tensors {}",
                e
            ),
        }
    }
    match fs::create_dir_all(output_dir_path) {
        Ok(_) => {}
        Err(e) => eprintln!("Export Error: Failed to create output directory: {}", e),
    };

    // check for existence, clear and create
    if tensor_dir_path.is_dir() {
        match fs::remove_dir_all(tensor_dir_path) {
            Ok(_) => {}
            Err(e) => eprintln!(
                "Export Error: Failed to clear out current output directory for tensors {}",
                e
            ),
        }
    }
    match fs::create_dir_all(tensor_dir_path) {
        Ok(_) => {}
        Err(e) => eprintln!("Export Error: Failed to create output directory: {}", e),
    };

    // check for existence, clear and create
    if node_dir_path.is_dir() {
        match fs::remove_dir_all(node_dir_path) {
            Ok(_) => {}
            Err(e) => eprintln!(
                "Export Error: Failed to clear out current output directory for tensors {}",
                e
            ),
        }
    }
    match fs::create_dir_all(node_dir_path) {
        Ok(_) => {}
        Err(e) => eprintln!("Export Error: Failed to create output directory: {}", e),
    };

    // create node registry and populate graph
    let mut node_registry: HashMap<*const (), String> = HashMap::new();
    let mut adjacency_list: Vec<(String, String)> = vec![];
    let mut tensor_registry: HashMap<*const (), String> = HashMap::new();

    let autograd_ref = tensor.get_autograd_ref();
    let root = autograd_ref
        .as_ref()
        .expect("Autograd does not exist")
        .get_grad_fn();

    if let Some(root) = root.as_ref() {
        populate_and_record_tensors_and_nodes(
            Rc::clone(root),
            &mut node_registry,
            &mut adjacency_list,
            &mut tensor_registry,
            &root_dir,
        );

        if let Ok(graph_json_str) = to_string_pretty(&adjacency_list) {
            let file = File::create(root_dir + "/graph_acyclic.json")
                .expect("Error: Creating graph.json for serializing graph failed");
            let mut writer = BufWriter::new(file);
            match writer.write_all(graph_json_str.as_bytes()) {
                Ok(_) => {}
                Err(e) => {
                    println!("{}", e);
                }
            }
        }
    } else {
        panic!("Export Error: Cannot export graph on tensor that does not allow gradient tracking");
    }
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
