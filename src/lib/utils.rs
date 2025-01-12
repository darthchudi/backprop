use std::collections::HashMap;
use std::fmt;
use crate::value::{Value, build_topological_graph};
use std::ops::{Mul, Div};


/// Generates a GraphViz DOT format string for the computation graph
/// rooted at `value`.
pub fn to_dot_string<T>(value: &Value<T>) -> String
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    let topo = build_topological_graph(value);

    // 2) Assign each node an integer ID for labeling
    let mut id_map = HashMap::new();
    for (i, node) in topo.iter().enumerate() {
        id_map.insert(node.borrow().id.clone(), i);
    }

    // 3) Start building the DOT string
    let mut output = String::new();
    output.push_str("digraph G {\n");
    output.push_str("  rankdir=\"LR\";\n");

    // 4) For each node in the topological order, create:
    //    - A node label showing data, gradient, operation and id
    //    - Edges from each ancestor -> this node
    for (i, node) in topo.iter().enumerate() {
        let inner = node.borrow();

        // Build a label for this node.
        let label = format!(
            "data={} | grad={:.4} | operation={} |id={}",
            inner.data,
            inner.gradient,
            inner.operation.to_str(),
            inner.id,
        );

        // Create the node line, e.g.:  N0 [label="data=5 | grad=0.00 | ..."];
        output.push_str(&format!("  N{} [shape=record, label=\"{}\"];\n", i, label));

        // For each ancestor, create an edge: ancestor -> node
        for ancestor in &inner.ancestors {
            let anc_id = id_map[&ancestor.borrow().id.clone()];
            
            // Draw arrow ancestor -> current node
            output.push_str(&format!("  N{} -> N{};\n", anc_id, i));
        }
    }

    output.push_str("}\n");
    output
}

pub fn write_graphiz_dot_file<T>(value: &Value<T>, output_name: &'static str)
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    let dot_str = to_dot_string(value);
    std::fs::write(output_name, dot_str).unwrap();
}

#[cfg(test)]
mod tests {
    use crate::value::{Value};
    use crate::utils::{write_graphiz_dot_file};
    
    #[test]
    fn render_topological_graph() {
        let a = &Value::new(4.0);
        let b = &Value::new(2.0);

        // Perform chained operations
        let c = a + b;       // c = a + b = 6
        let d = &c * b;       // d = c * b = 12
        let z = &d / a;       // z = d / a = 3

        write_graphiz_dot_file(&z, "graph.dot");
    }
}