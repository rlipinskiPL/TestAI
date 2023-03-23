package org.example.machine.learning.models.linear;

import org.example.machine.learning.activation.LinearFunction;
import org.example.machine.learning.loss.MeanSquaredError;

public class AdaptiveLinearNeuronBGD extends BatchGradientDescentModel {
    public AdaptiveLinearNeuronBGD(double learningRate, int epochs, int seed) {
        super(new LinearFunction(), new MeanSquaredError(), learningRate, epochs, seed);
    }

    public AdaptiveLinearNeuronBGD(double learningRate, int epochs) {
        super(new LinearFunction(), new MeanSquaredError(), learningRate, epochs);
    }
}
