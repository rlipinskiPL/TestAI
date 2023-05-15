package ai.test.machine.learning.models;

import ai.test.algebra.Shape;
import ai.test.machine.learning.layers.Layer;
import ai.test.machine.learning.loss.LossFunction;
import lombok.Getter;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public abstract class Model {

    protected List<Layer> layers = new ArrayList<>();

    /**
     * List of costs in particular epochs
     */
    @Getter
    protected List<Double> costs = new LinkedList<>();

    protected LossFunction lossFunction;

    protected double learningRate;

    protected boolean isCompiled = false;

    protected Model(LossFunction lossFunction, double learningRate) {
        this.lossFunction = lossFunction;
        this.learningRate = learningRate;
    }

    /**
     * Add new layer to the top of model layer stack.
     *
     * @param layer new layer
     */
    public abstract void addLayer(Layer layer);

    /**
     * Compile all layers in model.
     *
     * @param inputShape expected shape of input data
     */
    public abstract void compile(Shape inputShape);

    /**
     * This method trains the model based on the provided data
     *
     * @param input     training data
     * @param output    training labels
     * @param batchSize size of batch transferred to single training iteration
     * @param epochs    number of epochs
     * @param shuffle   if model should shuffle training data after each epoch
     */
    public abstract void fit(Object input, Object output, int batchSize, int epochs, boolean shuffle);

    /**
     * This method trains model with default values for: <br>
     * batch size is set to 32 <br>
     * epochs is set to 1 <br>
     * shuffle is set to false
     *
     * @param input  training data
     * @param output training labels
     */
    public void fit(Object input, Object output) {
        this.fit(input, output, 32, 1, false);
    }

    /**
     * This method predicts object basing on given data
     *
     * @param x input data
     * @return predicted object
     */
    public abstract Object predict(Object x);
}
