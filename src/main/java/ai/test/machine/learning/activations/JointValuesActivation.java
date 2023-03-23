package ai.test.machine.learning.activations;

import ai.test.algebra.Tensor;

//This class inform us that activation function compute output based on all outputs not only one as normal activation function
public abstract class JointValuesActivation implements ActivationFunction {
    @Override
    public double call(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }

    @Override
    public abstract Tensor derivative(Tensor X);

    public abstract Tensor callJointly(Tensor X);
}
