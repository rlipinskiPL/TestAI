package org.example.machine.learning.models.linear;

import org.example.algebra.Tensor;
import org.example.algebra.Matrix;
import org.example.algebra.Vector;
import org.example.machine.learning.activation.ActivationFunction;
import org.example.machine.learning.loss.LossFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public abstract class LinearModel {
    protected double learningRate;
    protected final int epochs;
    protected final int seed;
    protected List<Double> weights;
    protected double bias;
    protected List<Double> costs = new ArrayList<>();
    protected double[] classes;
    private final ActivationFunction activationFunction;
    private final LossFunction lossFunction;

    protected LinearModel(ActivationFunction activationFunction, LossFunction lossFunction, double learningRate, int epochs, int seed) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.seed = seed;
        this.activationFunction = activationFunction;
        this.lossFunction = lossFunction;
    }

    protected LinearModel(ActivationFunction activationFunction, LossFunction lossFunction, double learningRate, int epochs) {
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.seed = 1;
        this.activationFunction = activationFunction;
        this.lossFunction = lossFunction;
    }

    public abstract void fit(Matrix X, Vector Y);

    public double predict(Vector X) {
        double result = activation(netInput(X)).getAsScalar();
        return Math.abs(result-classes[0]) < Math.abs(result-classes[1]) ? classes[0] : classes[1];
    }

    public Vector predict(Matrix X) {
        List<Double> results = new ArrayList<>();
        for (int i = 0; i < X.height(); i++) {
            results.add(predict((Vector) X.getRow(i)));
        }
        return (Vector) Tensor.build(results, false);
    }

    public List<Double> getCosts() {
        return costs;
    }

    protected Tensor netInput(Tensor X) {
        return X.dot(Tensor.build(weights, false)).addition(bias);
    }

    protected Tensor activation(Tensor X) {
        return activationFunction.call(X);
    }

    //protected double loss(Tensor output, Tensor y){
    //    return lossFunction.call(output,y);
    //}

    protected double loss(double output, double y){
        return lossFunction.call(output, y);
    }

    protected void initParameters(Matrix X, Vector Y){
        weights = new ArrayList<>();
        costs = new ArrayList<>();

        Random rand = new Random(seed);
        for (int i = 0; i < X.width(); i++) {
            weights.add(rand.nextGaussian());
        }
        bias = rand.nextGaussian();

        double[] classes = Y.stream().mapToDouble(d->d).distinct().toArray();
        if(classes.length!=2){
            throw new IllegalStateException("This model is binary classification model and number of classes passed in vector Y is "+classes.length);
        }
        this.classes = classes;
    }
}
