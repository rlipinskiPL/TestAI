package ai.test.machine.learning.models;

import ai.test.algebra.Shape;
import ai.test.machine.learning.layers.Layer;
import ai.test.machine.learning.loss.LossFunction;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public abstract class Model {
    protected List<Layer> layers = new ArrayList<>();
    protected List<Double> costs = new LinkedList<>();
    protected LossFunction lossFunction;
    protected double learningRate;
    protected boolean isCompiled = false;

    protected Model(LossFunction lossFunction, double learningRate) {
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    public abstract void addLayer(Layer layer);

    public abstract void compile(Shape inputShape);

    public abstract void fit(Object input, Object output, int batchSize, int epochs, boolean shuffle);

    public void fit(Object input, Object output) {
        this.fit(input, output, 32, 1, false);
    }

    public abstract Object predict(Object x);

    public List<Double> getCosts() {
        return costs;
    }
}
