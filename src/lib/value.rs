use std::ops::{Add, Sub, Mul, Div};
use std::fmt;

// Value represents a numeric type used for mathematical operations
// It tracks and stores the gradients for operations which occur on the Value
#[derive(Debug, Clone, Copy)]
pub struct Value<T> {
    pub data: T,
}

impl<T: Copy> Value<T> {
    pub fn new(data: T) -> Value<T> {
        Value { data }
    }

    pub fn new_from_ref(data: &T) -> Value<T> {
        Value { data: *data }
    }

    pub fn val(&self) -> T {
        self.data
    }
}

impl<T: Add<Output=T>> Add for Value<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Value{data: self.data + rhs.data}
    }
}

impl<T: Sub<Output=T>> Sub for Value<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Value{data: self.data - rhs.data}
    }
}

impl<T: Mul<Output=T>> Mul for Value<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Value{data: self.data * rhs.data}
    }
}

impl<T: Div<Output=T>> Div for Value<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Value{data: self.data / rhs.data}
    }
}

// Allows the value to be printed via {}
impl<T: fmt::Display> fmt::Display for Value<T>{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

#[cfg(test)]
mod tests {
    use crate::value;

    #[test]
    fn arithmetic_ops_on_values(){
        let a = value::Value{data: 1.0};
        let b = value::Value{data: 2.0};

        let addition_op = a + b;
        assert_eq!(addition_op.data, 3.0);

        let subtraction_op = a - b;
        assert_eq!(subtraction_op.data, -1.0);

        let multiplication_op = a * b;
        assert_eq!(multiplication_op.data, 2.0);

        let division_op = a / b;
        assert_eq!(division_op.data, 0.5);
    }

    #[test]
    fn large_muls_on_values(){
        // 2^32
        let num = 2_f64.powi(32);

        let a: value::Value<f64> = value::Value{data: num};

        let c = a * a;

        assert!(c.data > 0.0);
    }
}