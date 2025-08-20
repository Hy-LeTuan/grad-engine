# GradEngine

**GradEngine** _(crate: grad_engine)_ is a Rust based implementation of an **automatic differentiation engine** coupled with a **computational graph visualizer**.

The project aims to implement the core components that power modern neural network training while also providing users a special viewpoint into how different components play together to form a coherent system.

# Table of Contents

<!--toc-->

-   [Quickstart](#quick-start)
-   [More About GradEngine](#more-about-gradengine)
    -   [My Inspiration](#inspiration)
    -   [Engineering Blog Series](#accompanied-blogs)
-   [Installation](#installation)
    -   [Prerequisites](#prerequisites)
        -   [Tensor Library](#rust-library)
        -   [Visualizer](#visualizer)
    -   [Getting GradEngine Source](#getting-gradengine-source)
    -   [Install Dependencies](#install-dependencies)
        -   [Crate Dependencies](#install-crate-dependencies)
        -   [Animation Dependencies](#install-animation-component-dependencies)
-   [Usage](#usage)
    -   [Running Examples](#running-examples)
    -   [Interacting With Tensors](#interacting-with-tensors)
        -   [Tensor Creation](#tensor-creation)
        -   [Tensor Operation](#tensor-operation)
        -   [Invoking Backpropagation and Graph Export](#invoking-backpropagation-and-graph-export)
    -   [Visualization](#visualization)
        -   [Compact Visualization](#compact-visualization)
        -   [Full Visualization](#full-visualization)
-   [License](#license)

<!-- tocstop -->

# Quickstart

For a quick visualization of the computation graph, you can run the **small exmaple** to generate a small computation graph

```bash
git clone https://github.com/Hy-LeTuan/grad-engine.git
cd grad-engine

# -- export will export graph at /output
cargo run --example small -- export
cd animation

manim -pqh main.py CreateAcyclicGraph
```

which will produce and play video of the full creation and execution process of the computation graph.

# More about GradEngine

## Inspiration

Greatly inspired by [PyTorch](https://github.com/pytorch/pytorch)’s design, GradEngine follows a dynamic computation graph generation design, and while simplified compared to production systems, it shows the essential mechanics of backpropagation and dynamic graph construction through the usage of _Computatoin Nodes_.

## Accompanied blogs

GradEngine started as a part of my deeper dive into the internals of modern Deep Learning frameworks, aiming to build both understanding and tooling by learning from popular highly optimized frameworks.

If you are interested in how the engine is laid out from scratch, GradEngine will be accompanied with a series of blogs explaining the internal mathematics of automatic differentiation engine as well as the engineering choices I've made. This blog series will be avaialble on my [personal portfolio](https://tuanhy-le.dev/blogs).

![Computation Graph with Mathematics](/docs/static/math-graph.png)

# Installation

The project has 2 separate components, which requires 2 different sets of prerequisites and installation process. The components and by extend the prerequisites, can be installed and run indepently of each other.

## Prerequisites

### Rust library

The `grad_engine` crate requires:

1. Rust edition 2024 (rustc 1.85.0 or later)
2. Cargo edition 2024 (cargo 1.85.0 or later)

### Visualizer

The Visualizer located in `animation` requires:

1. Python 3.11 or later
2. [Manim](https://github.com/ManimCommunity/manim) 0.19 or later if you are installing it separately from `pip`
3. `Manim` dependencies which you can refer to `Manim`'s guidelines to install:
    1. CMake
    2. pkgconfig
    3. pangocairo

## Getting GradEngine Source

```bash
git clone https://github.com/Hy-LeTuan/grad-engine
cd grad-engine
```

## Install Dependencies

### Install Crate Dependencies

```bash
cargo build

# to test functionalities of all backward nodes
cargo test
```

### Install Animation Component Dependencies

Before installing `animation`I recommend creating a virtual environment first. In local environment, I used `conda` to create the virtual environment, but `manim` suggests using `uv`, which you can install [here](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# manim is included in requirements.txt
cd animation
pip install -r requirements.txt

# if you're using uv
uv pip install -r requirements.txt
```

# Usage

The project has 2 components, offering you vastly different features:

1. The `grad_engine` library, located in `/src` gives you access to tensor creation, tensor operations, computation graph execution and computation graph export
2. The visualizer, located in `/animation`, allows you to create animation on how the computation graph is created and executed based on `JSON` exports of the computation graphs.

## Running Examples

I have built out some example computation graph in `/examples`, which you can run with

```bash
# cargo run --example [example_name]
cargo run --example small
cargo run --example large

# export args to export computation graph to /output
cargo run --example small -- export
cargo run --example large -- export
```

and the computation graph will be exported to their signified location. You will also see a terminal-based visualization of the graph displaying right after running these commands.

Visualize these examples through

```bash
cd animation
manim -pqh main.py CreateAcyclicGraph # for high quality render
manim -pql main.py CreateAcyclicGraph # for low quality render
```

## Interacting with tensors

### Tensor Creation

Create any tensor through the `tensor!` macro

```rust
// requires_grad will determine whether tensor is added to the computation graph or not

// creating a 1D tensor
let tensor = tensor!(1, 2, 3, 4, 5; requires_grad=true);

// creating a 2D tensor
let tensor = tensor!([1, 2, 3], [4, 5, 6]; requires_grad=false);

// creating a 3D tensor
let tensor = tensor!([[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]; requires_grad=true);

```

### Tensor Operations

GradEngine offers a variety of tensor operations, which can either be invoked directly through operations like `+`, `-`, through functions like `matmul` or through methods like `ln`.

```rust
use grad_engine::tensor;
use grad_engine::ops::public_ops::matmul::matmul;

let x1 = tensor!(1, 2, 3, 4, 5; requires_grad=true);
let x2 = tensor!(0.1, 0.2, 0.3, 0.4, 0.5);

// references have to be used if you don't want z to own x1 and x2
let z1 = &x1 + &x2.ln();

// matrix multiplication
let z2 = matmul(&x1, &x2);
```

### Invoking Backpropagation and Graph Export

Any tensor that is not a leaf tensor can invoke the `backward()` method to start the backpropagation process. The computation graph will be computed in this process, and will retain after `backward()` is complete for exportation.

```rust
use grad_engine::tensor;
use grad_engine::ops::public_ops::matmul::matmul;
use grad_engine::tensor_core::tensor::Tensor;

let x1 = tensor!(1, 2, 3, 4, 5; requires_grad=true);
let x2 = tensor!(0.1, 0.2, 0.3, 0.4, 0.5);

// matrix multiplication
let z = matmul(&x1, &x2);

// second parameter set to `true` to retain computation graph
z.backward(Tensor::ones_like(&z, None), true);

// export the graph through calling export_graph_acyclic, the graph will be stored in /output
export_graph_acyclic(&z, None);
```

## Visualization

I made 2 modes of visualizing the comptuation graph, one compact visualization which is a top down visualization of the graph directly in the terminal, and one full visualization which involves creating the animation in `animation`.

### Compact visualization

Visualize directly in the command line

```rust
use grad_engine::tensor;
use grad_engine::graph::visualize::visualizer::Visualizer;
use grad_engine::graph::visualize::visualizer::VisualizerTrait;
...
z.backward(Tensor::ones_like(&z, None), true);
Visualizer::visualize_graph(&z);
```

and in the case of the `large.rs` example, the visualized graph would look something like this

```bash
## Backward computation graph ##
------------------------------

AddBackward [ 2 child nodes ]
├── LnBackward [ 1 child nodes ]
    └── SubBackward [ 2 child nodes ]
        ├── MatmulBackward [ 2 child nodes ]
            ├── AddBackward [ 1 child nodes ]
                └── GradAccum [ Gradient accumulation ]
            └── SubBackward [ 2 child nodes ]
                ├── GradAccum [ Gradient accumulation ]
                └── GradAccum [ Gradient accumulation ]
        └── GradAccum [ Gradient accumulation ]
└── ExpBackward [ 1 child nodes ]
    └── GradAccum [ Gradient accumulation ]

------------------------------
```

### Full visualization

Creating the animation of the graph is a bit more complicated, as you'd have to use the `manim` interface.

```bash
cd animation
# CreateAcyclicGraph is the scene name
manim -pqh main.py CreateAcyclicGraph
```

will read the graph stored at `output/` and compute the animation. If your graph is stored somewhere else, you'd have to modify it in `animation/graph/parse_utils.py/`. The animation will be stored in `animation/media/videos/main/1080p60` and you can then play it on any `mp4` player.

**Animation from running _large_ example:**

[![Title](/docs/static/frame-from-computation-graph-anim.png)](/docs/static//CreateAcyclicGraph.mp4)

# License

[License: MIT](https://choosealicense.com/licenses/mit/)

![Rust 2024](https://img.shields.io/badge/rust-2024-orange)
