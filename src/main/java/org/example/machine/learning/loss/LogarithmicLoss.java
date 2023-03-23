package org.example.machine.learning.loss;

public class LogarithmicLoss implements LossFunction{

    @Override
    public double call(double x, double y) {
        return -(y*Math.log(x)+(1-y)*Math.log(1-x));
    }

    @Override
    public double derivative(double x, double y) {
        return (1-y)/(1-x) - y/x;
    }
}
