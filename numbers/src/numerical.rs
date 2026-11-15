use std::ops::{Add, Mul};

use crate::arith::{Magnitude, Scalar};

pub struct Tolerance<T> {
    pub(crate) relative: T,
    pub(crate) absolute: T,
}

impl<T> Tolerance<T> {
    pub fn with_relative(self, tol: T) -> Self {
        Self {
            relative: tol,
            ..self
        }
    }

    pub fn with_absolute(self, tol: T) -> Self {
        Self {
            absolute: tol,
            ..self
        }
    }
}

macro_rules! default_tolerance {
    ($type:ty) => {
        impl Default for Tolerance<$type> {
            fn default() -> Self {
                Self {
                    relative: (10000 as $type).recip(),
                    absolute: (100000000 as $type).recip(),
                }
            }
        }
    };
}

default_tolerance!(f64);
default_tolerance!(f32);

// TODO: can not have to type <T as Magnitude>::Output everywhere
pub fn is_close<T>(a: T, b: T, tolerance: Tolerance<<T as Magnitude>::Output>) -> bool
where
    T: Scalar + Magnitude,
    <T as Magnitude>::Output:
        Add<Output = <T as Magnitude>::Output> + Mul<Output = <T as Magnitude>::Output>,
{
    let rhs = tolerance.absolute + tolerance.relative * b.abs();
    let lhs = (a + b.neg()).abs();

    lhs <= rhs
}
