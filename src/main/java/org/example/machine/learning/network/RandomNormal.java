package org.example.machine.learning.network;

import org.example.algebra.Tensor;

import java.util.Random;

public class RandomNormal implements Initializer{

    private final int seed;
    private final Random rand;

    public RandomNormal(int seed) {
        this.seed = seed;
        rand = new Random(System.currentTimeMillis());
    }

    @Override
    public double compute(Tensor tensor) {
        for (int i = 0; i < tensor.height(); i++) {
            for(int j = 0; j < tensor.width(); j++){
                tensor.set(i,j,rand.nextGaussian(0,0.1));
            }
        }
        return rand.nextGaussian();
    }
}
