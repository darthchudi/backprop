use std::ops::{Add, Sub, Mul, Div, DerefMut, Deref};
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

impl<T: Copy> Value<T> {
    pub fn new(data: T) -> Value<T> {
        let inner_value = InnerValue {
            data,
            gradient: 0.0,
            ancestors: vec![],
            symbol: "",
        };

        Value(Rc::new(RefCell::new(inner_value)))
    }

    pub fn backward(&self) {
        let val = self.borrow();

        match val.symbol{
            "+" => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];
                

                left_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
                right_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
                
                println!("Addition backward pass!")
            },
            "-" => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                left_ancestor.borrow_mut().gradient += 1.0 * val.gradient;
                right_ancestor.borrow_mut().gradient += 1.0 * val.gradient;

                println!("Subtraction backward pass!")
            }
            "*" => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];

                left_ancestor.borrow_mut().gradient += right_ancestor.borrow().data * val.gradient;
                right_ancestor.borrow_mut().gradient += left_ancestor.borrow().data * val.gradient;

                println!("Subtraction backward pass!")
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
}

impl<T: Add<Output=T> + Copy + 'static> Add for Value<T> {
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

impl<T: Add<Output=T> + Copy + 'static> Add for &Value<T> {
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


impl<T: Sub<Output=T> + Copy + 'static> Sub for Value<T> {
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


impl<T: Sub<Output=T> + Copy + 'static> Sub for &Value<T> {
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


//
// impl<T: Mul<Output=T>> Mul for Value<T> {
//     type Output = Self;
//
//     fn mul(self, rhs: Self) -> Self::Output {
//         let result = self.data * rhs.data;
//
//         Value::new(result)
//     }
// }
//
// impl<T: Div<Output=T>> Div for Value<T> {
//     type Output = Self;
//
//     fn div(self, rhs: Self) -> Self::Output {
//         let result = self.data / rhs.data;
//
//         Value::new(result)
//     }
// }
//

// Allows the value to be printed via {}
// impl<T: fmt::Display> fmt::Display for Value<T>{
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.data)
//     }
// }
//
// impl<T: fmt::Debug> fmt::Debug for Value<T>{
//     fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
//         write!(f, "{}", self.data)
//     }
// }

#[cfg(test)]
mod tests {
    use crate::value;
    use crate::value::{InnerValue, Value};
    use std::rc::Rc;

    #[test]
    fn simple_addition_on_values(){
        // Initial addition
        let x = Value::new(1);
        let w = Value::new(2);

        // Get a reference of x to assert on the gradient later
        let x_clone = x.clone();

        let y = x + w;
        assert_eq!(y.get_data(), 3);

        // Get a reference to y to perform the backward pass
        let y_clone = y.clone();

        // Subsequent addition
        let z = y + Value::new(10);
        assert_eq!(z.get_data(), 13);

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
        let x = &Value::new(1);
        let w = &Value::new(2);

        let y = x + w;

        let z = &y + &Value::new(10);

        z.set_gradient(1.0);

        z.backward();
        y.backward();

        println!("⭐️ {:?}", z);
        println!("⭐️ {:?}", y);

        println!("dz/dx = {}", x.get_gradient());

        assert_eq!(x.get_gradient(), 1.0);
        assert_eq!(y.get_gradient(), 1.0);
    }

    #[test]
    fn simple_subtraction_on_values(){
        let x = Value::new(11);
        let w = Value::new(2);

        let x_1 = x.clone();

        let y = x - w;
        
        assert_eq!(y.get_data(), 9);

        println!("⭐️ {:?}", y);

        y.set_gradient(1.0);
        y.backward();

        println!("⭐️ {:?}", y);

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

        println!("⭐️ {:?}", z);
        println!("⭐️ {:?}", y);

        println!("dz/dx = {}", x.get_gradient());

        assert_eq!(x.get_gradient(), 1.0);
        assert_eq!(y.get_gradient(), 1.0);
    }

    // #[test]
    // fn large_muls_on_values(){
    //     // 2^32
    //     let num = 2_f64.powi(32);
    //
    //     let a: value::Value<f64> = value::Value::new(num);
    //
    //     let c = a * a;
    //
    //     assert!(c.data > 0.0);
    // }
}