use crate::{
    Matrix, Vector,
    arith::{Scalar, TryMul},
    err::LinAlgError,
};

impl<T: Scalar + Clone> TryMul<Vector<T>> for Matrix<T> {
    type Error = LinAlgError;
    type Output = Vector<T>;

    fn try_mul(self, other: Vector<T>) -> Result<Self::Output, Self::Error> {
        let (m, n) = self.shape();
        if n != other.shape() {
            return Err(LinAlgError::IncompatibleShape);
        }

        let mut result = Vec::new();
        let column_vector: Matrix<T> = other.into();
        for i in 0..m {
            // TODO: there has to be a cleaner way to do this
            let row = self.row(i);
            let product = row.try_mul(column_vector.clone())?;
            result.push(product[(0, 0)].clone());
        }

        Ok(result.into())
    }
}

pub fn solve<T: Scalar>(a: Matrix<T>, b: Vector<T>) -> Result<Vector<T>, LinAlgError> {
    if !a.is_square() {
        return Err(LinAlgError::IncompatibleShape);
    }

    let (n, _) = a.shape();
    if n == 0 || n != b.shape() {
        return Err(LinAlgError::IncompatibleShape);
    }

    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::try_mul;

    #[test]
    fn test_try_mul() {
        let expected = Vector::from([5.0, 11.0]);
        let matrix = Matrix::try_from_iter([[1.0, 2.0], [3.0, 4.0]]).unwrap();
        let vector = Vector::from([1.0, 2.0]);
        let actual = try_mul(matrix, vector).unwrap();

        assert_eq!(expected, actual);
    }
}
