package ai.test.machine.learning.regularizers;

import ai.test.algebra.Tensor;

public interface Regularizer {
    Tensor computeDerivative(Tensor weights);
}
