package ai.test.machine.learning.activations;

public class LinearFunction implements ActivationFunction {

    @Override
    public double call(double x) {
        return x;
    }

    @Override
    public double derivative(double x) {
        return 1;
    }
}
