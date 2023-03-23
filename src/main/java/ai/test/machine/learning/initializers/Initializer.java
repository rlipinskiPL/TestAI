package ai.test.machine.learning.initializers;

import ai.test.algebra.Tensor;

public interface Initializer {
    double compute(Tensor tensor);
}
