package ai.test.machine.learning.activations;

public class SigmoidFunction implements ActivationFunction {

    @Override
    public double call(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivative(double x) {
        return Math.exp(x) / Math.pow(Math.exp(x) + 1, 2);
    }
}
