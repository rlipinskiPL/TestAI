package ai.test.machine.learning.activations;

import ai.test.algebra.Tensor;

/**
 * This interface specifies two required methods for each activation function.
 * We assume that all functions that directly implement this interface are scalar function not vector function.
 * This assumption allow us to create two default methods that make computations for Tensor object, to be more precise
 * defaults methods make computation for each element from tensor independently.
 */
public interface ActivationFunction {

    /**
     * This method compute output of function for all elements in tensor.
     *
     * @param X input tensor to activation function
     * @return output of function
     */
    default Tensor call(Tensor X) {
        double[][] result = new double[X.height()][X.width()];
        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                result[i][j] = call(X.get(i, j));
            }
        }
        return Tensor.build(result);
    }

    /**
     * This method compute derivative of function for all elements in tensor.
     *
     * @param X input tensor to activation function
     * @return derivative of function
     */
    default Tensor derivative(Tensor X) {
        double[][] result = new double[X.height()][X.width()];
        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                result[i][j] = derivative(X.get(i, j));
            }
        }
        return Tensor.build(result);
    }

    /**
     * This method compute activation function output.
     *
     * @param x input value to activation function
     * @return output of function
     */
    double call(double x);

    /**
     * This method compute activation function derivative in given point.
     *
     * @param x input value to activation function
     * @return derivative of function
     */
    double derivative(double x);
}
