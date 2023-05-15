package ai.test.machine.learning.activations;

import ai.test.algebra.Tensor;

/**
 * Assumption in ActivationFunction interface says that all functions that directly implements this interface must be scalar function.
 * This class represents all vector functions, so it doesn't allow to use methods with scalar parameter.
 * It overrides two default functions from ActivationFunction interface, because now it is not available to make
 * computations independently for all elements as now we have to analise input data as sequence of vectors.
 */
public abstract class JointValuesActivation implements ActivationFunction {

    @Override
    public double call(double x) {
        throw new UnsupportedOperationException("JointValueActivation represents vector function so it doesn't support method with scalar parameter.");
    }

    @Override
    public double derivative(double x) {
        throw new UnsupportedOperationException("JointValueActivation represents vector function so it doesn't support method with scalar parameter.");

    }

    /**
     * @param X input tensor to activation function
     * @return jacobi matrix of vector function
     */
    @Override
    public abstract Tensor derivative(Tensor X);

    /**
     * @param X input tensor to activation function
     * @return output of vector function
     */
    @Override
    public abstract Tensor call(Tensor X);
}
