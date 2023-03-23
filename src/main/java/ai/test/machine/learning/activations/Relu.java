package ai.test.machine.learning.activations;

public class Relu implements ActivationFunction {
    @Override
    public double call(double x) {
        return x > 0 ? x : 0;
    }

    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }
}
