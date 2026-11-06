use std::ops::{Add, Mul};

use num::{Float, Signed, Zero};

// element of a field
pub trait Scalar: Sized + Add + Mul + Zero {
    fn neg(&self) -> Self;
    fn inverse(&self) -> Option<Self>;
}

impl<T> Scalar for T
where
    T: Float + Signed,
{
    fn neg(&self) -> Self {
        -*self
    }

    fn inverse(&self) -> Option<Self> {
        if *self == Self::zero() || *self == Self::neg_zero() {
            None
        } else {
            Some(Self::one() / *self)
        }
    }
}

pub trait OrderedScalar: Scalar + Ord {}

impl<T> OrderedScalar for T where T: Scalar + Ord {}

pub trait TryAdd<Rhs = Self> {
    type Error;
    type Output;

    fn try_add(self, other: Rhs) -> Result<Self::Output, Self::Error>;
}

pub fn try_add<Rhs, Lhs: TryAdd<Rhs>>(left: Lhs, right: Rhs) -> Result<Lhs::Output, Lhs::Error> {
    left.try_add(right)
}
