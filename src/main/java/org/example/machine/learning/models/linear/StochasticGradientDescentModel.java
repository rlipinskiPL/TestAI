package org.example.machine.learning.models.linear;

import org.example.algebra.Matrix;
import org.example.algebra.Vector;
import org.example.machine.learning.activation.ActivationFunction;
import org.example.machine.learning.loss.LossFunction;
import org.example.machine.learning.models.linear.LinearModel;

import java.util.ArrayList;
import java.util.List;

public class StochasticGradientDescentModel extends LinearModel {

    protected StochasticGradientDescentModel(ActivationFunction activationFunction, LossFunction lossFunction, double learningRate, int epochs, int seed) {
        super(activationFunction, lossFunction, learningRate, epochs, seed);
    }

    protected StochasticGradientDescentModel(ActivationFunction activationFunction, LossFunction lossFunction, double learningRate, int epochs) {
        super(activationFunction, lossFunction, learningRate, epochs);
    }

    @Override
    public void fit(Matrix X, Vector Y) {
        initParameters(X,Y);

        for(int i=0;i<epochs;i++){
            //ToDo shuffle
            List<Double> cost = new ArrayList<>();
            for(int j=0;j<X.height();j++){
                cost.add(updateWeights((Vector) X.getRow(j), Y.get(j)));
            }
            costs.add(cost.stream().mapToDouble(Double::doubleValue).sum());
        }
    }

    public void partialFit(Matrix X, Vector Y){
        if(weights == null){
            fit(X,Y);
            return;
        }

        for(int i=0;i<epochs;i++){
            //ToDo shuffle
            List<Double> cost = new ArrayList<>();
            for(int j=0;j<X.height();j++){
                cost.add(updateWeights((Vector) X.getRow(j), Y.get(j)));
            }
            costs.add(cost.stream().mapToDouble(Double::doubleValue).average().getAsDouble());
        }
    }

    private double updateWeights(Vector X, double y){
        double output = activation(netInput(X)).getAsScalar();
        double error = y - output;
        Vector update = (Vector) X.multiply(error*learningRate);
        for(int i=0;i<weights.size();i++){
            weights.set(i,weights.get(i)+update.get(i));
        }
        bias+=learningRate*error;
        return loss(output,y);
    }
}
