use std::{convert::identity, iter::{self, zip}, ops::{Add, Div, Mul}};

use num::{Signed, Zero};

// TODO: the num traits are a shorthand
pub trait Field: Signed + Zero + Add + Mul {}

impl<T> Field for T where T: Signed + Zero + Add + Mul {}

pub trait TrySum<Rhs = Self> {
    type Error;
    type Output;

    fn try_sum(self, other: Rhs) -> Result<Self::Output, Self::Error>;
}

pub trait ArrayLike {
    type Dtype: Field;

    fn shape(&self) -> Vec<usize>;
}

impl<T: Field> ArrayLike for Vec<T> {
    type Dtype = T;
    
    fn shape(&self) -> Vec<usize> {
        return [self.len()].to_vec()
    }
}

pub enum LinAlgError {
    IncompatibleShape,
}

impl<T: Field, I: Iterator<Item = T>> TrySum for I {
    type Error = LinAlgError;
    type Output = Vec<T>;

    fn try_sum(mut self, mut other: Self) -> Result<Self::Output, Self::Error> {
        let mut sum = Vec::new();
        while let (Some(a), Some(b)) = (self.next(), other.next()) {
            sum.push(a + b);
        };

        if self.next().is_some() || other.next().is_some() {
            return Err(LinAlgError::IncompatibleShape)
        }

        Ok(sum)
    }
}

struct Matrix<T> {
    vec: Vec<T>,
    stride: usize,
}

impl<T: Zero> Matrix<T> {
    fn zeros<S: Into<(usize, usize)>>(shape: S) -> Self {
        let (rows, cols) = shape.into();
        let len = rows * cols;
        let vec = iter::repeat_with(|| T::zero()).take(len).collect();
        Matrix {
            vec, stride: cols,
        }
    }
}

impl<T> TryFrom<Vec<Vec<T>>> for Matrix<T> {
    type Error = LinAlgError;

    fn try_from(value: Vec<Vec<T>>) -> Result<Self, Self::Error> {
        if value.is_empty() {
            return Ok(Matrix {
                vec: Vec::new(),
                stride: 0,
            })
        }
        
        let stride = value[0].len();
        if !value.iter().all(|v| v.len() == stride) {
            return Err(LinAlgError::IncompatibleShape)
        }

        let vec = value.into_iter().flat_map(identity).collect();
        Ok(Matrix {
            vec, stride
        })
    }
}

impl<T: Field> ArrayLike for Matrix<T> {
    type Dtype = T;

    fn shape(&self) -> Vec<usize> {
        [self.vec.len() / self.stride, self.stride].to_vec()
    }
}

impl<T: Add<Output = T>> TrySum for Matrix<T> {
    type Error = LinAlgError;
    type Output = Self;

    fn try_sum(self, other: Self) -> Result<Self::Output, Self::Error> {
        if self.stride != other.stride {
            return Err(LinAlgError::IncompatibleShape)
        }
        
        let mut sum = Vec::new();
        let mut self_iter = self.vec.into_iter();
        let mut other_iter = other.vec.into_iter();

        while let (Some(a), Some(b)) = (self_iter.next(), other_iter.next()) {
            sum.push(a + b);
        };

        if self_iter.next().is_some() || other_iter.next().is_some() {
            return Err(LinAlgError::IncompatibleShape)
        }

        Ok(Matrix {
            vec: sum,
            stride: self.stride,
        })
    }
}