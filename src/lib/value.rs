use std::ops::{Add, Sub, Mul, Div, DerefMut, Deref};
use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;



// BackwardPassFn represents a function which when called on a value, computes
// the differential of the value relative to its inputs.
// type BackwardPassFn = dyn FnOnce() -> Result<(), String>;

// Value represents a numeric type used for mathematical operations
// It tracks and stores the gradients for operations which occur on the Value
// pub struct Value<T> {
//     pub data: T,
//     children: Vec<Rc<Value<T>>>,
//     gradient: f64,
//     _backward: Option<Box<BackwardPassFn>>,
//     symbol: &'static str,
// }

// impl<T> Clone for Value<T> {
//     fn clone(&self) -> Self {
//         Self {
//             data: *self.data,
//             children: self.children.clone(),
//             gradient: 0.0,
//             _backward: None,
//             symbol: "",
//         }
//     }
// }


#[derive(Clone)]
pub struct CloneableFn {
    inner: Rc<Box<dyn FnMut()>>,
}

impl CloneableFn {
    pub fn new<F: FnMut() + 'static>(f: F) -> Self {
        Self {
            inner: Rc::new(Box::new(f)),
        }
    }
}

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
    pub ancestors: Vec<Value<T>>,

    pub ancestors_rc: Vec<Rc<Value<T>>>,

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
            .field("ancestors_rc", &self.ancestors_rc)
            .field("gradient", &self.gradient)
            .field("symbol", &self.symbol)
            .finish()
    }
}

// Value is a tuple struct which wraps an InnerValue
#[derive(Debug, Clone)]
pub struct Value<T>(RefCell<InnerValue<T>>);

impl<T> Deref for Value<T> {
    type Target = RefCell<InnerValue<T>>;

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
            ancestors_rc: vec![],
            symbol: "",
        };

        Value(RefCell::new(inner_value))
    }

    pub fn backward(&self) {
        let val = self.borrow();

        match val.symbol{
            "+" => {
                let left_ancestor = &val.ancestors[0];
                let right_ancestor = &val.ancestors[1];
                
                println!("{}", left_ancestor.borrow().gradient);

                left_ancestor.borrow_mut().gradient += 1.0;
                right_ancestor.borrow_mut().gradient += 1.0;

                println!("{}", left_ancestor.borrow().gradient);

                println!("Added!")
            },
            _ => ()
        }

    }
}


// impl<T: Copy> Value<T> {
//     pub fn new(data: T) -> Value<T> {
//         Value { data, children: vec![], gradient: 0.0, _backward: None, symbol: ""}
//     }
//
//     pub fn new_from_ref(data: &T) -> Value<T> {
//         Value { data: *data, children: vec![], gradient: 0.0, _backward: None, symbol: "" }
//     }
//
//     pub fn val(&self) -> T {
//         self.data
//     }
// }

// impl<T: Add<Output=T>> Add for Value<T> {
//     type Output = Self;
//
//     fn add(self, rhs: Self) -> Self::Output {
//         let result =  self.data + rhs.data;
//         let mut value = Value::new(result);
//
//         value.children.append(vec![Rc::clone(self), Rc::clone(rhs)]);
//
//         let backward = || {
//
//         };
//     }
// }

impl<T: Add<Output=T> + Copy + 'static> Add for Value<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let result =  self.borrow().data + rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![self.clone(), rhs.clone()];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "+";

        let ancestors_clone = value.borrow().ancestors.clone();

        let left_ancestor = value.borrow().ancestors[0].clone();
        let right_ancestor = value.borrow().ancestors[0].clone();

        let _ = move || {
            left_ancestor.borrow_mut().gradient += 1.0;
            right_ancestor.borrow_mut().gradient += 1.0;
        };

    //    value.borrow_mut()._backward = CloneableFn::new(backward);

       value
    }
}

impl<'a, T: Add<Output=T> + Copy + 'static> Add for &'a Value<T> {
    type Output = Value<T>;

    fn add(self, rhs:  Self) -> Self::Output {
        let result =  self.borrow().data + rhs.borrow().data;
        let value = Value::new(result);

        // Set a reference to the ancestors
        let mut ancestors = vec![self.clone(), rhs.clone()];
        value.borrow_mut().ancestors.append(&mut ancestors);

        value.borrow_mut().symbol = "+";

        let ancestors_clone = value.borrow().ancestors.clone();

        let left_ancestor = value.borrow().ancestors[0].clone();
        let right_ancestor = value.borrow().ancestors[0].clone();

        let _ = move || {
            left_ancestor.borrow_mut().gradient += 1.0;
            right_ancestor.borrow_mut().gradient += 1.0;
        };

        //    value.borrow_mut()._backward = CloneableFn::new(backward);

        value
    }
}

//
// impl<T: Sub<Output=T>> Sub for Value<T> {
//     type Output = Self;
//
//     fn sub(self, rhs: Self) -> Self::Output {
//         let result =  self.data - rhs.data;
//
//         Value::new(result)
//     }
// }
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
    fn arithmetic_ops_on_values(){
        let x = &Value::new(1);
        let w = &Value::new(2);
        
        let y = x + w;

        println!("⭐️ {:?}", y);

        y.backward();

        println!("⭐️ {:?}", y);
        
        println!("dy/dx = {}", x.borrow().gradient)


        // let a = &value::Value::new(1.0);
        // let b = &value::Value::new(2.0);
        //
        // let addition_op = a + b;
        // assert_eq!(addition_op.data, 3.0);
        //
        // let subtraction_op = a - b;
        // assert_eq!(subtraction_op.data, -1.0);
        //
        // let multiplication_op = a * b;
        // assert_eq!(multiplication_op.data, 2.0);
        //
        // let division_op = a / b;
        // assert_eq!(division_op.data, 0.5);
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