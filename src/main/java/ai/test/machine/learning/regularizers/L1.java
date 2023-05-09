package ai.test.machine.learning.regularizers;

import ai.test.algebra.Tensor;

public class L1 implements Regularizer {

    public final double alpha;


    public L1(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public Tensor computeDerivative(Tensor weights) {
        double[][] result = new double[weights.height()][weights.width()];
        double sum = 0;

        for (int i = 0; i < weights.height(); i++) {
            for (int j = 0; j < weights.width(); j++) {
                sum += weights.get(i, j);
            }
        }

        double sign = Math.signum(sum);
        for (int i = 0; i < weights.height(); i++) {
            for (int j = 0; j < weights.width(); j++) {
                result[i][j] = alpha * sign;
            }
        }

        return Tensor.build(result);
    }
}
