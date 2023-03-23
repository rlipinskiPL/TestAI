package org.example.machine.learning.models.linear;

import org.example.algebra.Matrix;
import org.example.algebra.Vector;
import org.example.machine.learning.activation.LinearFunction;
import org.example.machine.learning.models.linear.LinearModel;

public class Perceptron extends LinearModel {
    public Perceptron(double learningRate, int epochs, int seed) {
        super(new LinearFunction(), null, learningRate, epochs, seed);
    }

    public Perceptron(double learningRate, int epochs) {
        super(new LinearFunction(), null, learningRate, epochs);
    }

    @Override
    public void fit(Matrix X, Vector Y) {
        initParameters(X,Y);

        for (int i = 0; i < epochs; i++) {
            int error = 0;
            for (int j = 0; j < X.height(); j++) {
                Vector currentRow = (Vector) X.getRow(j);
                double update = learningRate * (Y.get(j, 0) - predict(currentRow));
                for (int k = 0; k < weights.size(); k++) {
                    weights.set(k, weights.get(k) + update * currentRow.get(0, k));
                }
                bias += update;
                error += update == 0 ? 0 : 1;
            }
            costs.add((double) error);
        }
    }
}
