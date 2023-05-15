package ai.test.machine.learning.loss;

/**
 * This class is a singleton class for all loss functions.
 */
public class Loss {

    private static final LossFunction cce = new CategoricalCrossEntropy();

    private static final LossFunction logarithmic = new LogarithmicLoss();

    private static final LossFunction mse = new MeanSquaredError();

    /**
     * @return categorical cross-entropy function
     */
    public static LossFunction categoricalCrossEntropy() {
        return cce;
    }

    /**
     * @return logarithmic loss function
     */
    public static LossFunction logarithmicLoss() {
        return logarithmic;
    }

    /**
     * @return mean squared error function
     */
    public static LossFunction meanSquaredError() {
        return mse;
    }
}
