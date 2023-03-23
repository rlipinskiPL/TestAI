package org.example.machine.learning.loss;

public class CategoricalCrossEntropy implements LossFunction {
    @Override
    public double call(double x, double y) {
        return -y * Math.log(x + Math.pow(10, -100));
    }

    @Override
    public double derivative(double x, double y) {
        return -y / (x + Math.pow(10, -100));
    }
}
