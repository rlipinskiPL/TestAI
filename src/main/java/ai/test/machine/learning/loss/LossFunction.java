package ai.test.machine.learning.loss;

import ai.test.algebra.Tensor;

/**
 * This interface specifies two required methods for each loss function.
 * We assume that all functions that directly implement this interface are scalar function not vector function.
 * This assumption allow us to create two default methods that make computations for Tensor object, to be more precise
 * defaults methods make computation for each element from tensor independently.
 */
public interface LossFunction {

    /**
     * This method computes loss function output for all elements in tensor.
     *
     * @param X predicted values
     * @param Y true values
     * @return output of function
     */
    default Tensor call(Tensor X, Tensor Y) {
        double[][] result = new double[X.height()][X.width()];
        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                result[i][j] = call(X.get(i, j), Y.get(i, j));
            }
        }
        return Tensor.build(result);
    }

    /**
     * This method computes loss function derivative for all elements in tensor.
     *
     * @param X predicted values
     * @param Y true values
     * @return derivative of loss function
     */
    default Tensor derivative(Tensor X, Tensor Y) {
        double[][] result = new double[X.height()][X.width()];
        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                result[i][j] = derivative(X.get(i, j), Y.get(i, j));
            }
        }
        return Tensor.build(result);
    }

    /**
     * This method computes loss function output.
     *
     * @param x predicted value
     * @param y true value
     * @return output of loss function
     */
    double call(double x, double y);

    /**
     * This method computes loss function derivative of loss function in given point.
     *
     * @param x predicted value
     * @param y true value
     * @return derivative of loss function
     */
    double derivative(double x, double y);
}
