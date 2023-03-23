package org.example.machine.learning.loss;

public class MeanSquaredError implements LossFunction{
    @Override
    public double call(double x, double y) {
        return 0.5*Math.pow((y-x),2);
    }

    @Override
    public double derivative(double x, double y) {
        return x-y;
    }
}
