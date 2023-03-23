package org.example.machine.learning.models.linear;

import org.example.algebra.Vector;
import org.example.machine.learning.activation.SigmoidFunction;
import org.example.machine.learning.loss.LogarithmicLoss;

public class LogisticRegressionSGD extends StochasticGradientDescentModel {

    public LogisticRegressionSGD(double learningRate, int epochs, int seed) {
        super(new SigmoidFunction(), new LogarithmicLoss(), learningRate, epochs, seed);
    }

    public LogisticRegressionSGD(double learningRate, int epochs) {
        super(new SigmoidFunction(), new LogarithmicLoss(), learningRate, epochs);
    }

    @Override
    public double predict(Vector X){
        double result = activation(netInput(X)).getAsScalar();
        return result < 0.5 ? classes[0] : classes[1];
    }
}
