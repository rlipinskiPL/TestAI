package ai.test.machine.learning.initializers;

import ai.test.algebra.Tensor;

import java.util.Random;

/**
 * This class initialize all hiperparameters randomly in accordance to
 * normal distribution with mean equals to 0 and standard deviation equals to 0.05.
 */
public class RandomNormal implements Initializer {

    private final Random rand;

    public RandomNormal() {
        rand = new Random(System.currentTimeMillis());
    }

    public RandomNormal(int seed) {
        rand = new Random(seed);
    }

    @Override
    public double call(Tensor tensor) {
        for (int i = 0; i < tensor.height(); i++) {
            for (int j = 0; j < tensor.width(); j++) {
                tensor.set(i, j, rand.nextGaussian() * 0.05);
            }
        }
        return rand.nextGaussian();
    }
}
