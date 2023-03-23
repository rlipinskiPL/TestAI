package ai.test.machine.learning.loss;

public class Loss {
    private static final LossFunction cce = new CategoricalCrossEntropy();
    private static final LossFunction logarithmic = new LogarithmicLoss();
    private static final LossFunction mse = new MeanSquaredError();

    public static LossFunction categoricalCrossEntropy(){
        return cce;
    }

    public static LossFunction logarithmicLoss(){
        return logarithmic;
    }

    public static LossFunction meanSquaredError(){
        return mse;
    }
}
