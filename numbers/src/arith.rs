use std::ops::{Add, Mul, Neg, Sub};

use num::{Float, One, Signed, Zero};

// element of a field
pub trait Scalar:
    Sized
    + Add<Output = Self>
    + Mul<Output = Self>
    + Zero
    + Neg<Output = Self>
    + PartialEq
    + One
    + Sub<Output = Self>
{
    fn inverse(&self) -> Option<Self>;
}

impl<T> Scalar for T
where
    T: Float + Signed,
{
    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            Some(self.recip())
        }
    }
}

pub trait OrderedScalar: Scalar + PartialOrd {}

impl<T> OrderedScalar for T where T: Scalar + PartialOrd {}

pub trait Magnitude {
    type Output: PartialOrd + Add<Output = Self::Output> + Mul<Output = Self::Output>;

    fn abs(&self) -> Self::Output;
}

impl<T> Magnitude for T
where
    T: Float,
{
    type Output = Self;

    fn abs(&self) -> Self::Output {
        Float::abs(*self)
    }
}

pub trait TryAdd<Rhs = Self> {
    type Error;
    type Output;

    fn try_add(self, other: Rhs) -> Result<Self::Output, Self::Error>;
}

pub fn try_add<Rhs, Lhs: TryAdd<Rhs>>(left: Lhs, right: Rhs) -> Result<Lhs::Output, Lhs::Error> {
    left.try_add(right)
}

pub trait TryMul<Rhs = Self> {
    type Error;
    type Output;

    fn try_mul(self, other: Rhs) -> Result<Self::Output, Self::Error>;
}

pub fn try_mul<Rhs, Lhs: TryMul<Rhs>>(left: Lhs, right: Rhs) -> Result<Lhs::Output, Lhs::Error> {
    left.try_mul(right)
}
