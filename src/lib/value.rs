use std::ops::{Add, Sub, Mul, Div, Deref};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use rand::Rng;

/// ValueOp represents an arithmetic operation that can be performed on 1 or more Value types.
#[derive(Debug, Clone)]
pub enum ValueOp {
    Addition,
    Subtraction,
    Multiplication,
    Division,
    None,
}


impl ValueOp {
    pub fn to_str(&self) -> &'static str {
        match self{
            ValueOp::Addition => "+",
            ValueOp::Subtraction => "-",
            ValueOp::Multiplication => "*",
            ValueOp::Division => "/",
            ValueOp::None => "none",
        } 
    }
}

/// InnerValue represents the inner contents of a Value object in a computation graph.
/// A given InnerValue will have references to the nodes which created it, which are referred to as ancestors.
/// By maintaining a reference to its ancestors and only generating the gradients when the backward pass is lazily
/// evaluated, a given value can propagate its own gradients backwards to its ancestors gradients when evaluated topologically.
/// For a given output y = w + x
/// where y is the output node, w = 10, x = 33; we'll get the following domain representation:
/// InnerValue.data = 20
/// InnerValue.ancestors = Vec<Rc<RefCell<InnerValue<10>>>, Rc<RefCell<InnerValue<10>>>>
/// InnerValue.gradient = 0
/// InnerValue.operation = Addition
#[derive(Clone)]
pub struct InnerValue<T> {
    // data represents the scalar value
    pub data: T,

    // ancestors refers to the values (nodes) which are passed as inputs to this node
    // the relationship might be inverted here for modelling reasons, which I'll be exploring further.
    pub ancestors: Vec<Rc<RefCell<InnerValue<T>>>>,

    // gradient is the gradient of this value relative to it's "parent" nodes
    // i.e for an equation y = 1 + x.
    // where x is the value.
    // dy/dx = 1
    // so x.gradient = 1
    // The gradient of a given node is relative to the value it creates in a given equation
    pub gradient: f64,

    pub operation: ValueOp,

    pub id: String,
}

impl<T: fmt::Debug> fmt::Debug for InnerValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InnerValue")
            .field("id", &self.id)
            .field("data", &self.data)
            .field("ancestors", &self.ancestors)
            .field("gradient", &self.gradient)
            .field("operation", &self.operation)
            .finish()
    }
}

impl<T: fmt::Display + fmt::Debug> fmt::Display for InnerValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InnerValue")
            .field("id", &self.id)
            .field("data", &self.data)
            .field("ancestors", &self.ancestors)
            .field("gradient", &self.gradient)
            .field("operation", &self.operation)
            .finish()
    }
}

/// Value is a tuple struct which wraps an InnerValue
/// It provides support for auto-differentiable mathematical operations.
#[derive(Debug, Clone)]
pub struct Value<T>(Rc<RefCell<InnerValue<T>>>);

impl<T: fmt::Display + fmt::Debug> fmt::Display for Value<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("innerValue", &self.0.borrow())
            .finish()
    }
}

impl<T> Deref for Value<T> {
    type Target = Rc<RefCell<InnerValue<T>>>;

    fn deref(&self) -> &Self::Target { 
        &self.0
    }
}

impl<T> Value<T>
where T: Copy + Mul<f64, Output = f64> + Div<T, Output = T> + Into<f64> + From<f64> + fmt::Display + fmt::Debug  + 'static
{
    fn generate_id() -> String {
        let mut rng = rand::thread_rng();

        let id_num: f64 = rng.gen_range(1.0.. 10000.0);
        let id = format!("valueid_{}", id_num).replace(".", ""); // replace the decimal point

        id.to_string()
    }

    pub fn new(data: T) -> Value<T> {
        let id = Self::generate_id();

        let inner_value = InnerValue {
            id,
            data,
            gradient: 0.0,
            ancestors: vec![],
            operation: ValueOp::None,
        };

        Value(Rc::new(RefCell::new(inner_value)))
    }

    pub fn new_from_ref(data: &T) -> Value<T> {
        Value::new(*data)
    }

    pub fn new_with_id(data: T, id: &'static str) -> Value<T> {
        let value = Value::new(data);
        value.borrow_mut().id = id.to_string();

        value
    }

    // backward computes the gradients for a given node value.
    // The ancestor nodes which generated this value will have their gradient values updated based on the derivate of the
    // given node relative to the ancestor.
    pub fn backward(&self) {
        let val = self.borrow();

        match val.operation{
            ValueOp::Addition => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                left_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
                right_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
            },
            ValueOp::Subtraction => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                left_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
                right_ancestor.borrow_mut().gradient -= 1.0 * val.gradient;
            }
            ValueOp::Multiplication=> {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                let left_ancestor_data: f64 = left_ancestor.borrow().data.into();
                let right_ancestor_data: f64 = right_ancestor.borrow().data.into();

                left_ancestor.borrow_mut().gradient += right_ancestor_data * val.gradient;
                right_ancestor.borrow_mut().gradient += left_ancestor_data * val.gradient;
            }
            ValueOp::Division => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                let left_ancestor_data: f64 = left_ancestor.borrow().data.into();
                let right_ancestor_data: f64 = right_ancestor.borrow().data.into();

                left_ancestor.borrow_mut().gradient += (1.0/right_ancestor_data) * val.gradient;
                right_ancestor.borrow_mut().gradient -= (left_ancestor_data/(right_ancestor_data * right_ancestor_data)) * val.gradient;
            }
            _ => ()
        }
    }

    /// run_grad builds a topological graph of computations and then performs the backpropagation algorithm
    /// to update the derivatives of nodes in the computation graph.
    /// The given node is taken as the start node from which the dependencies in the graph are built.
    pub fn run_grad(&self){
        // Set the initial gradient for the root node.
        self.set_gradient(1.0);

        let topological_graph = build_topological_graph(self);

        // reverse the topological graph because we want the computed gradients to flow backwards to ancestors
        let reversed_topological_graph: Vec<&Rc<RefCell<InnerValue<T>>>> = topological_graph.iter().rev().collect();

        // Compute the gradient for nodes in the graph
        for node in reversed_topological_graph {
            let node_as_value = Value(Rc::clone(node));
            node_as_value.backward();
        }
    }

    pub fn get_data(&self) -> T {
        self.borrow().data
    }

    pub fn get_gradient(&self) -> f64 {
        self.borrow().gradient
    }

    pub fn set_gradient(&self, gradient: f64) {
        self.borrow_mut().gradient = gradient;
    }

    pub fn get_id(&self) -> String {
        self.borrow().id.clone()
    }

    pub fn clear_gradient(&self) {
        self.borrow_mut().gradient = 0.0;

        // todo: clear gradients on ancestors before removing references
        self.borrow_mut().ancestors.clear();
    }
}

/// order_nodes_topologically builds a topological order for nodes based on their dependencies.
pub fn build_topological_graph<T>(value: &Value<T>) -> Vec<Rc<RefCell<InnerValue<T>>>>
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64> + fmt::Display + fmt::Debug {
    let mut seen_nodes: HashMap<String, bool> = HashMap::new();

    order_nodes_topologically(value, &mut seen_nodes)
}

/// order_nodes_topologically returns a topologically ordered set of ancestor nodes for a given node.
fn order_nodes_topologically<T>(value: &Value<T>, seen_nodes: &mut HashMap<String, bool>) -> Vec<Rc<RefCell<InnerValue<T>>>>
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    let mut nodes = vec![];

    let value_id = value.get_id();

    if seen_nodes.contains_key(&value_id) {
        return nodes;
    }

    // Mark the node as seen incase its referenced in any of its own ancestors.
    seen_nodes.insert(value_id, true);

    for ancestor in value.borrow().ancestors.iter(){
        // Process the dependencies for this ancestor node
        let ancestor_as_value = Value(Rc::clone(ancestor));
        let mut ancestor_dependencies = order_nodes_topologically(&ancestor_as_value, seen_nodes);

        // Add the ancestor's dependencies to the list
        nodes.append(&mut ancestor_dependencies);
    }

    // Add the node to the list after processing its ancestors
    nodes.push(Rc::clone(value));

    nodes
}

pub fn print_topological_graph<T>(topological_graph: Vec<Rc<RefCell<InnerValue<T>>>>)
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    for item in topological_graph {
        println!("{}", item.borrow());
    }
}
impl<T> Add for Value<T>
where T: Add<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Div<T, Output = T> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data + rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().operation = ValueOp::Addition;
        
        value
    }
}

impl<T> Add for &Value<T>
where T: Add<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Div<T, Output = T> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    type Output = Value<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data + rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(self), Rc::clone(rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().operation = ValueOp::Addition;

        value
    }
}


impl<T> Sub for Value<T>
where T: Sub<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + Div<T, Output = T> + From<f64> + fmt::Display + fmt::Debug
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data - rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().operation = ValueOp::Subtraction;

        value
    }
}


impl<T> Sub for &Value<T>
where T: Sub<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Div<T, Output = T> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    type Output = Value<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data - rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(self), Rc::clone(rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().operation = ValueOp::Subtraction;

        value
    }
}

impl<T> Mul for Value<T>
where T: Mul<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Div<T, Output = T> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data * rhs.borrow().data;
        let value = Value::new(result);

        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];

        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().operation = ValueOp::Multiplication;

        value
    }
}

impl<T> Mul for &Value<T>
where T: Mul<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Div<T, Output = T> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    type Output = Value<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data * rhs.borrow().data;
        let value = Value::new(result);

        let mut ancestors = vec![Rc::clone(self), Rc::clone(rhs)];

        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().operation = ValueOp::Multiplication;

        value
    }
}

impl<T> Div for Value<T>
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data / rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().operation = ValueOp::Division;

        value
    }
}

impl<T> Div for &Value<T>
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64> + fmt::Display + fmt::Debug
{
    type Output = Value<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data / rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(self), Rc::clone(rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().operation = ValueOp::Division;

        value
    }
}

#[cfg(test)]
mod tests {
    use crate::value::{build_topological_graph, Value};

    #[test]
    fn simple_addition_on_values(){
        // Initial addition
        let x = Value::new(1.0);
        let w = Value::new(2.0);

        // Get a reference of x to assert on the gradient later
        let x_clone = x.clone();

        let y = x + w;
        assert_eq!(y.get_data(), 3.0);

        // Get a reference to y to perform the backward pass
        let y_clone = y.clone();

        // Subsequent addition
        let z = y + Value::new(10.0);
        assert_eq!(z.get_data(), 13.0);

        // Set the gradient for z and calculate the gradient
        // We start from z as it's the root node.
        z.run_grad();

        assert_eq!(x_clone.borrow().gradient, 1.0);
        assert_eq!(y_clone.borrow().gradient, 1.0);
    }

    #[test]
    fn addition_references(){
        let x = &Value::new(1.0);
        let w = &Value::new(2.0);

        let y = x + w;

        let z = &y + &Value::new(10.0);

        z.run_grad();

        assert_eq!(x.get_gradient(), 1.0);
        assert_eq!(y.get_gradient(), 1.0);
    }

    #[test]
    fn simple_subtraction_on_values(){
        let x = Value::new(11.0);
        let w = Value::new(2.0);

        let x_1 = x.clone();

        let y = x - w;

        assert_eq!(y.get_data(), 9.0);

        y.run_grad();

        assert_eq!(x_1.get_gradient(), 1.0);
    }

    #[test]
    fn subtraction_references(){
        let x = &Value::new(50.5);
        let w = &Value::new(20.0);

        let y = x - w;

        let z = &y - &Value::new(0.5);
        assert_eq!(z.get_data(), 30.0);

       z.run_grad();
        
        assert_eq!(x.get_gradient(), 1.0);
        assert_eq!(y.get_gradient(), 1.0);
    }

    #[test]
    fn simple_multiplication_on_values(){
        let x = Value::new(10.0);
        let w = Value::new( 20.0);

        let x_1 = x.clone();
        let w_1 = w.clone();

        let y = x * w;
        
        y.run_grad();

        assert_eq!(x_1.get_gradient(), 20.0);
        assert_eq!(w_1.get_gradient(), 10.0);
    }

    #[test]
    fn multiplication_references(){
        let x = &Value::new(10.0);
        let w = &Value::new( 20.0);

        let y = x * w;

        assert_eq!(y.get_data(), 200.0);
        assert_eq!(x.get_gradient(), 0.0);

        y.run_grad();
        
        assert_eq!(x.get_gradient(), 20.0);
        assert_eq!(w.get_gradient(), 10.0);
    }

    #[test]
    fn simple_div_on_values(){
        let x = Value::new(100.0);
        let w = Value::new( 2.0);

        let x_1 = x.clone();
        let w_1 = w.clone();

        let y = x / w;

        assert_eq!(y.get_data(), 50.0);
        assert_eq!(x_1.get_gradient(), 0.0);

        y.run_grad();
        
        assert_eq!(x_1.get_gradient(), 0.5);
        assert_eq!(w_1.get_gradient(), -25.0);
    }

    #[test]
    fn div_on_references(){
        let x = &Value::new(100.0);
        let w = &Value::new( 2.0);

        let y = x / w;

        assert_eq!(y.get_data(), 50.0);
        assert_eq!(x.get_gradient(), 0.0);

        y.run_grad();

        assert_eq!(x.get_gradient(), 0.5);
        assert_eq!(w.get_gradient(), -25.0);
    }

    #[test]
    fn test_backward_pass_chained_operations() {
        let a = Value::new(4.0);
        let b = Value::new(2.0);

        // Perform chained operations
        let c = a.clone() + b.clone();       // c = a + b = 6
        let d = c.clone() * b.clone();       // d = c * b = 12
        let z = d.clone() / a.clone();       // z = d / a = 3
        
        z.run_grad();

        assert_eq!(a.borrow().gradient, -0.25, "Expected a.gradient to be -0.25");
        assert_eq!(b.borrow().gradient, 2.0, "Expected b.gradient to be 2.0");
    }

    #[test]
    fn test_build_topological_graph() {
        let a = Value::new(4.0);
        let b = Value::new(2.0);

        // Perform chained operations
        let c = a.clone() + b.clone();       // c = a + b = 6
        let d = c.clone() * b.clone();       // d = c * b = 12
        let z = d.clone() / a.clone();       // z = d / a = 3

        let topological_graph = build_topological_graph(&z);

        let expected_order = vec![4.0, 2.0, 6.0, 12.0, 3.0];

        let actual_order: Vec<f64> = topological_graph.iter().map(|n| n.borrow().data).collect();
        
        assert_eq!(expected_order, actual_order);
    }

    #[test]
    fn test_backward_pass_harder_case() {
        // Create initial values
        let a = Value::new_with_id(2.0, "a");
        let b = Value::new_with_id(3.0, "b");

        // Perform chained operations
        let c = a.clone() + b.clone();               // c = a + b
        c.borrow_mut().id = "c".to_string();


        let d = c.clone() * (b.clone() - a.clone());  // d = c * (b - a)
        d.borrow_mut().id = "d".to_string();

        let z = d.clone() / b.clone();               // z = d / b
        z.borrow_mut().id = "z".to_string();

        // Forward pass checks
        assert_eq!(c.borrow().data, 5.0, "Expected c.data to be 5.0");
        assert_eq!(d.borrow().data, 5.0, "Expected d.data to be 5.0");
        assert_eq!(z.borrow().data, 5.0 / 3.0, "Expected z.data to be 5.0 / 3.0");

        z.run_grad();

        // Backward pass checks
        assert_eq!(z.borrow().gradient, 1.0, "z.gradient should be 1.0");
        assert_eq!(round_to_places(c.borrow().gradient, 2), 0.33, "c.gradient should be 0.33");
        assert_eq!(round_to_places(b.borrow().gradient, 2), 1.44, "b.gradient should be 1.44");
        assert_eq!(round_to_places(a.borrow().gradient, 2), -1.33, "a.gradient should be −1.33");
    }

    fn round_to_places(value: f64, places: u32) -> f64 {
        let factor = 10f64.powi(places as i32);
        (value * factor).round() / factor
    }
}

