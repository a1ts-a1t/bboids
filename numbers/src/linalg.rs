use crate::{
    Matrix, Vector,
    arith::{Scalar, TryMul},
    err::LinAlgError,
    vector,
};

impl<T: Scalar + Clone> TryMul<Vector<T>> for Matrix<T> {
    type Error = LinAlgError;
    type Output = Vector<T>;

    fn try_mul(self, other: Vector<T>) -> Result<Self::Output, Self::Error> {
        let n = self.shape().1;
        if n != other.shape() {
            return Err(LinAlgError::IncompatibleShape);
        }

        let mut result = Vec::new();
        for row in self.rows() {
            let row_vec: Vector<T> = Vector::from(row);
            result.push(vector::dot(row_vec, &other)?);
        }

        Ok(result.into())
    }
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
