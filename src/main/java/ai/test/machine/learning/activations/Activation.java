package ai.test.machine.learning.activations;

/**
 * This class is singleton class for all activation functions.
 */
public class Activation {

    private static final ActivationFunction linear = new LinearFunction();

    private static final ActivationFunction relu = new Relu();

    private static final ActivationFunction sigmoid = new SigmoidFunction();

    private static final ActivationFunction softMax = new SoftMax();

    /**
     * @return linear function
     */
    public static ActivationFunction linearFunction() {
        return linear;
    }

    /**
     * @return relu function
     */
    public static ActivationFunction relu() {
        return relu;
    }

    /**
     * @return sigmoid function
     */
    public static ActivationFunction sigmoidFunction() {
        return sigmoid;
    }

    /**
     * @return softmax function
     */
    public static ActivationFunction softMax() {
        return softMax;
    }
}
