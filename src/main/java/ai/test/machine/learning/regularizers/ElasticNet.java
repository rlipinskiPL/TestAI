package ai.test.machine.learning.regularizers;

import ai.test.algebra.Tensor;

/**
 * This class implements Regularizer interface by using elastic net algorithm.
 * In short, it is a combination of the L1 and L2 algorithm.
 */
public class ElasticNet implements Regularizer {

    private final L1 l1;

    private final L2 l2;

    public ElasticNet(double l1, double l2) {
        this.l1 = new L1(l1);
        this.l2 = new L2(l2);
    }

    @Override
    public Tensor computeDerivative(Tensor weights) {
        return l1.computeDerivative(weights).add(l2.computeDerivative(weights));
    }
}
