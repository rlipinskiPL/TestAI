package org.example.machine.learning.activation;

import org.example.algebra.Tensor;

public abstract class JointValuesActivation implements ActivationFunction{
    @Override
    public double call(double x) {
        return x;
    }

    @Override
    public double derivative(double x){
        return 1;
    }

    @Override
    public abstract Tensor derivative(Tensor X);

    public abstract Tensor callJointly(Tensor X);
}
