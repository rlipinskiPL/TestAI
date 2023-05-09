package ai.test.machine.learning.activations;

public class Activation {

    private static final ActivationFunction linear = new LinearFunction();

    private static final ActivationFunction relu = new Relu();

    private static final ActivationFunction sigmoid = new SigmoidFunction();

    private static final ActivationFunction softMax = new SoftMax();

    public static ActivationFunction linearFunction() {
        return linear;
    }

    public static ActivationFunction relu() {
        return relu;
    }

    public static ActivationFunction sigmoidFunction() {
        return sigmoid;
    }

    public static ActivationFunction softMax() {
        return softMax;
    }
}
