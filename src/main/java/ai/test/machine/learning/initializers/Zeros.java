package ai.test.machine.learning.initializers;

import ai.test.algebra.Tensor;

/**
 * This class initalize all hiperparameters to 0.
 */
public class Zeros implements Initializer {
    @Override
    public double call(Tensor tensor) {
        for (int i = 0; i < tensor.height(); i++) {
            for (int j = 0; j < tensor.width(); j++) {
                tensor.set(i, j, 0);
            }
        }
        return 0;
    }
}
