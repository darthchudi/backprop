## Backprop

Backprop is an [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) implementation in Rust. 

It supports computing the gradients of nodes in a computation graph for basic math operations: `addition`, `subtraction`, `multiplication` and `division`.

### Example usage 

```rust
use backprop::{value};

fn main(){
    let x = &Value::new(1.0);
    let w = &Value::new(2.0);
    
    let y = x + w;
    
    // Computes gradients for the computation graph
    y.backward();
}
```

### Running tests

```shell
cargo test
```

### Todo

- [] Replace dynamically allocated `Vec<f64>` with `[f64]` arrays.
- [] Visualise computation graph using `.dot` file from GraphViz
- [] Update Value ID to use randomly generated ID
- [] Add Benchmarks

### Acknowledgments

1. This library was inspired by [Kaparthy's micrograd Python library](https://github.com/karpathy/micrograd).
2. The [Rust documentation on Smart pointers](https://doc.rust-lang.org/book/ch15-00-smart-pointers.html) was really helpful in modelling the nodes within the graph.