use std::ops::{Add, Sub, Mul, Div, Deref};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;

// InnerValue represents the inner contents of a Value object in a computation graph.
// A given InnerValue will have references to the nodes which created it.
// By maintaining a reference to its ancestors and only generating the gradients when the backward
// pass is lazily evaluated, a given value can propagate its own gradients backwards to its
// ancestors gradients when evaluated topologically.
// For a given output y = w + x
// where y is the output node, w = 10, x = 33; we'll get the following domain representation:
// InnerValue.data = 20
// InnerValue.ancestors = Vec<Rc<Value<10>>, Rc<Value<33>>>
// InnerValue.gradient = 0
// InnerValue._backward.
// InnerValue.symbol = "+"
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
    // Todo: Explore if this means a given value needs to be strictly owned in our computation graph
    pub gradient: f64,
    
    pub symbol: &'static str,
}

impl<T: fmt::Debug> fmt::Debug for InnerValue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InnerValue")
            .field("data", &self.data)
            .field("ancestors", &self.ancestors)
            .field("gradient", &self.gradient)
            .field("symbol", &self.symbol)
            .finish()
    }
}

// Value is a tuple struct which wraps an InnerValue
#[derive(Debug, Clone)]
pub struct Value<T>(Rc<RefCell<InnerValue<T>>>);

impl<T> Deref for Value<T> {
    type Target = Rc<RefCell<InnerValue<T>>>;

    fn deref(&self) -> &Self::Target { 
        &self.0
    }
}

impl<T> Value<T>
where T: Copy + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    pub fn new(data: T) -> Value<T> {
        let inner_value = InnerValue {
            data,
            gradient: 0.0,
            ancestors: vec![],
            symbol: "",
        };

        Value(Rc::new(RefCell::new(inner_value)))
    }

    pub fn new_from_ref(data: &T) -> Value<T> {
        Value::new(*data)
    }

    pub fn backward(&self) {
        let val = self.borrow();
        
        match val.symbol{
            "+" => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                left_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
                right_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
            },
            "-" => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                left_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
                right_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
            }
            "*" => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                left_ancestor.borrow_mut().gradient += right_ancestor.borrow().data * val.gradient;
                right_ancestor.borrow_mut().gradient += left_ancestor.borrow().data * val.gradient;
            }
            "/" => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                let right_ancestor_data: f64 = right_ancestor.borrow().data.into();
                left_ancestor.borrow_mut().gradient += (1.0/right_ancestor_data) * val.gradient;
                
                let left_ancestor_data: f64 = left_ancestor.borrow().data.into();
                right_ancestor.borrow_mut().gradient += (-left_ancestor_data/(right_ancestor_data * right_ancestor_data)) * val.gradient;
            }
            _ => ()
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
    
    pub fn clear_gradient(&self) {
        self.borrow_mut().gradient = 0.0;
        self.borrow_mut().ancestors.clear();
    }
}

impl<T> Add for Value<T>
where T: Add<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data + rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "+";
        
        value
    }
}

impl<T> Add for &Value<T>
where T: Add<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    type Output = Value<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data + rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "+";

        value
    }
}


impl<T> Sub for Value<T>
where T: Sub<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data - rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "-";

        value
    }
}


impl<T> Sub for &Value<T>
where T: Sub<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    type Output = Value<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data - rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "-";

        value
    }
}

impl<T> Mul for Value<T>
where T: Mul<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data * rhs.borrow().data;
        let value = Value::new(result);

        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];

        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "*";

        value
    }
}

impl<T> Mul for &Value<T>
where T: Mul<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    type Output = Value<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data * rhs.borrow().data;
        let value = Value::new(result);

        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];

        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "*";

        value
    }
}

impl<T> Div for Value<T>
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data / rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "/";

        value
    }
}

impl<T> Div for &Value<T>
where T: Div<Output=T> + Copy + 'static + Mul<f64, Output = f64> + Into<f64> + From<f64>
{
    type Output = Value<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data / rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![Rc::clone(&self), Rc::clone(&rhs)];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "/";

        value
    }
}

#[cfg(test)]
mod tests {
    use crate::value::{Value};

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
        z.set_gradient(1.0);

        // Todo call in topological order
        z.backward();
        y_clone.backward();

        assert_eq!(x_clone.borrow().gradient, 1.0);
        assert_eq!(y_clone.borrow().gradient, 1.0);
    }

    #[test]
    fn addition_references(){
        let x = &Value::new(1.0);
        let w = &Value::new(2.0);

        let y = x + w;

        let z = &y + &Value::new(10.0);

        z.set_gradient(1.0);

        z.backward();
        y.backward();

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
        
        y.set_gradient(1.0);
        y.backward();
        
        assert_eq!(x_1.get_gradient(), 1.0);
    }

    #[test]
    fn subtraction_references(){
        let x = &Value::new(50.5);
        let w = &Value::new(20.0);

        let y = x - w;

        let z = &y - &Value::new(0.5);
        assert_eq!(z.get_data(), 30.0);

        z.set_gradient(1.0);

        z.backward();
        y.backward();
        
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

        y.set_gradient(1.0);
        y.backward();

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

        y.set_gradient(1.0);
        y.backward();
        
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

        y.set_gradient(1.0);
        y.backward();
        
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

        y.set_gradient(1.0);
        y.backward();
        
        assert_eq!(x.get_gradient(), 0.5);
        assert_eq!(w.get_gradient(), -25.0);
    }
}