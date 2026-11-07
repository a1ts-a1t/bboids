use std::{borrow::Borrow, iter::zip, ops::Add};

use crate::{
    arith::{Scalar, TryAdd},
    err::LinAlgError,
};

#[derive(PartialEq, Debug, Clone)]
pub struct Vector<T>(Vec<T>);

impl<T> Vector<T> {
    pub fn shape(&self) -> usize {
        self.0.len()
    }
}

impl<T> From<Vec<T>> for Vector<T> {
    fn from(value: Vec<T>) -> Self {
        Vector(value)
    }
}

impl<T: Clone> From<&[T]> for Vector<T> {
    fn from(value: &[T]) -> Self {
        Vector(value.to_vec())
    }
}

impl<const N: usize, T> From<[T; N]> for Vector<T> {
    fn from(value: [T; N]) -> Self {
        Vector(value.into())
    }
}

impl<T> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Vector(iter.into_iter().collect())
    }
}

impl<T> IntoIterator for Vector<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: Add<Output = T>> TryAdd for Vector<T> {
    type Error = LinAlgError;
    type Output = Self;

    fn try_add(self, other: Self) -> Result<Self::Output, Self::Error> {
        let mut self_iter = self.into_iter();
        let mut other_iter = other.into_iter();
        let mut sum = Vec::new();
        while let (Some(a), Some(b)) = (self_iter.next(), other_iter.next()) {
            sum.push(a + b);
        }

        if self_iter.next().is_some() || other_iter.next().is_some() {
            return Err(LinAlgError::IncompatibleShape);
        }

        Ok(Vector(sum))
    }
}

pub fn dot<T: Scalar + Clone, Lhs: Borrow<Vector<T>>, Rhs: Borrow<Vector<T>>>(
    lhs: Lhs,
    rhs: Rhs,
) -> Result<T, LinAlgError> {
    let vec1 = &lhs.borrow().0;
    let vec2 = &rhs.borrow().0;

    if vec1.len() != vec2.len() {
        return Err(LinAlgError::IncompatibleShape);
    }

    Ok(zip(vec1, vec2).fold(T::zero(), |acc, (a, b)| acc + (a.clone() * b.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let expected = 20.0;
        let actual = dot(Vector::from([1.0, 2.0, 3.0]), Vector::from([2.0, 3.0, 4.0])).unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_dot_fails() {
        let expected = LinAlgError::IncompatibleShape;
        let actual = dot(&Vector::from([1.0]), Vector::from([2.0, 3.0, 4.0])).unwrap_err();

        assert_eq!(expected, actual);
    }
}
