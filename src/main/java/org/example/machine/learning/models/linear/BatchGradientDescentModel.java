package org.example.machine.learning.models.linear;

import org.example.algebra.Tensor;
import org.example.algebra.Matrix;
import org.example.algebra.Vector;
import org.example.machine.learning.activation.ActivationFunction;
import org.example.machine.learning.loss.LossFunction;

public class BatchGradientDescentModel extends LinearModel {
    protected BatchGradientDescentModel(ActivationFunction activationFunction, LossFunction lossFunction, double learningRate, int epochs, int seed) {
        super(activationFunction, lossFunction, learningRate, epochs, seed);
    }

    protected BatchGradientDescentModel(ActivationFunction activationFunction, LossFunction lossFunction, double learningRate, int epochs) {
        super(activationFunction, lossFunction, learningRate, epochs);
    }

    @Override
    public void fit(Matrix X, Vector Y) {
        initParameters(X,Y);
        for (int i = 0; i < epochs; i++) {
            Tensor netInput = netInput(X);
            Tensor output = activation(netInput);
            Vector errors = (Vector) Y.sub(output);
            Vector updates = (Vector) X.transpose().dot(errors).multiply(learningRate);
            for (int j = 0; j < updates.height(); j++) {
                weights.set(j, weights.get(j) + updates.get(j));
            }
            bias += learningRate * errors.stream().mapToDouble(d->d).sum();
            //ToDo costs.add(loss(output,Y));
        }
    }
}
