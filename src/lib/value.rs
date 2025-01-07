use std::ops::{Add, Sub, Mul, Div};
use std::fmt;

// Value represents a numeric type used for mathematical operations
#[derive(Debug, Clone, Copy)]
pub struct Value<T> {
    pub value: T,
}

impl<T: Add<Output=T>> Add for Value<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Value{value: self.value + rhs.value}
    }
}

impl<T: Sub<Output=T>> Sub for Value<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Value{value: self.value - rhs.value}
    }
}

impl<T: Mul<Output=T>> Mul for Value<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Value{value: self.value * rhs.value}
    }
}

impl<T: Div<Output=T>> Div for Value<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Value{value: self.value / rhs.value}
    }
}

// Allows the value to be printed via {}
impl<T: fmt::Display> fmt::Display for Value<T>{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[cfg(test)]
mod tests {
    use crate::value;

    #[test]
    fn arithmetic_ops_on_values(){
        let a = value::Value{value: 1.0};
        let b = value::Value{value: 2.0};

        let addition_op = a + b;
        assert_eq!(addition_op.value, 3.0);

        let subtraction_op = a - b;
        assert_eq!(subtraction_op.value, -1.0);

        let multiplication_op = a * b;
        assert_eq!(multiplication_op.value, 2.0);

        let division_op = a / b;
        assert_eq!(division_op.value, 0.5);
    }

    #[test]
    fn large_muls_on_values(){
        // 2^32
        let num = 2_f64.powi(32);

        let a: value::Value<f64> = value::Value{value: num};

        let c = a * a;

        assert!(c.value > 0.0);
    }
}