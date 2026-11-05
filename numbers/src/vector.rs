use std::ops::Add;

use crate::{arith::TryAdd, err::LinAlgError};

struct Vector<T>(Vec<T>);

impl<T> IntoIterator for Vector<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: Add<Output = T>, I: IntoIterator<Item = T>> TryAdd for I {
    type Error = LinAlgError;
    type Output = Vec<T>;

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

        Ok(sum)
    }
}
