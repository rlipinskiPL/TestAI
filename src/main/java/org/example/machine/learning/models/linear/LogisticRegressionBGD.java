package org.example.machine.learning.models.linear;

import org.example.algebra.Vector;
import org.example.machine.learning.activation.SigmoidFunction;
import org.example.machine.learning.loss.LogarithmicLoss;
import org.example.machine.learning.models.linear.BatchGradientDescentModel;

public class LogisticRegressionBGD extends BatchGradientDescentModel {
    public LogisticRegressionBGD(double learningRate, int epochs, int seed) {
        super(new SigmoidFunction(), new LogarithmicLoss(), learningRate, epochs, seed);
    }

    public LogisticRegressionBGD(double learningRate, int epochs) {
        super(new SigmoidFunction(), new LogarithmicLoss(), learningRate, epochs);
    }

    @Override
    public double predict(Vector X){
        double result = activation(netInput(X)).getAsScalar();
        return result < 0.5 ? classes[0] : classes[1];
    }
}
