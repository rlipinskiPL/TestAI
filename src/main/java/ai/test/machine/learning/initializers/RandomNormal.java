package ai.test.machine.learning.initializers;

import ai.test.algebra.Tensor;

import java.util.Random;

public class RandomNormal implements Initializer {

    private final Random rand;

    public RandomNormal() {
        rand = new Random(System.currentTimeMillis());
    }

    public RandomNormal(int seed) {
        rand = new Random(seed);
    }

    //initialize values in tensor and return next random number(needed to init bias)
    @Override
    public double compute(Tensor tensor) {
        for (int i = 0; i < tensor.height(); i++) {
            for (int j = 0; j < tensor.width(); j++) {
                tensor.set(i, j, rand.nextGaussian());
            }
        }
        return rand.nextGaussian();
    }
}
