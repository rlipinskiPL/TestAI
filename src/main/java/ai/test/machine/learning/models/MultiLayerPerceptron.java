package ai.test.machine.learning.models;

import ai.test.algebra.Matrix;
import ai.test.algebra.Shape;
import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;
import ai.test.machine.learning.layers.DropOutLayer;
import ai.test.machine.learning.layers.Layer;
import ai.test.machine.learning.layers.MLPLayer;
import ai.test.machine.learning.loss.LossFunction;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.DoubleStream;

public class MultiLayerPerceptron extends Model {

    private Tensor lastInput;

    public MultiLayerPerceptron(LossFunction lossFunction, double learningRate) {
        super(lossFunction, learningRate);
    }

    @Override
    public void addLayer(Layer layer) {
        if (layer.getClass() != MLPLayer.class && layer.getClass() != DropOutLayer.class) {
            throw new IllegalArgumentException(layer.getClass().toString() + " is not acceptable in MultiLayerPerceptron (only MLPLayer and DropOutLayer)");
        }
        layers.add(layer);
    }

    @Override
    public void compile(Shape inputShape) {
        layers.get(0).compile(inputShape);
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).compile(layers.get(i - 1).getShape());
        }
        isCompiled = true;
    }

    @Override
    public void fit(Object input, Object output, int batchSize, int epochs, boolean shuffle) {
        if (!Tensor.class.isAssignableFrom(input.getClass()) && !Tensor.class.isAssignableFrom(output.getClass())) {
            throw new IllegalArgumentException("Only Tensors are acceptable as a data in MLP");
        }
        fit((Tensor) input, (Tensor) output, batchSize, epochs, shuffle);
    }

    public void fit(Tensor input, Tensor output, int batchSize, int epochs, boolean shuffle) {
        if (!isCompiled) {
            throw new IllegalStateException("Model is not compiled! Compile model before using it");
        } else if (batchSize < 0 || epochs < 0) {
            throw new IllegalArgumentException("Epochs and batchSize must be positive number");
        }
        for (int i = 0; i < epochs; i++) {
            if (shuffle) {
                //ToDo shuffle
            }
            int currentRow = 0;
            List<Double> currentCosts = new LinkedList<>();
            while (currentRow < input.height()) {
                Tensor X = input.cut(currentRow, currentRow + batchSize - 1);
                Tensor Y = output.cut(currentRow, currentRow + batchSize - 1);
                lastInput = X;

                Tensor netOutput = feedForward(X);

                Layer lastLayer = layers.get(layers.size() - 1);
                Tensor outputError = lossFunction.derivative(netOutput, Y).elementwise(lastLayer.getActivationFunction().derivative(lastLayer.getLastImpulse()));
                Tensor cost = lossFunction.call(netOutput, Y);
                if (cost.getClass() == Vector.class) {
                    currentCosts.add(((Vector) cost).stream().mapToDouble(d -> d).average().orElseThrow());
                } else {
                    currentCosts.add(((Matrix) cost).stream().flatMapToDouble(array -> DoubleStream.of(Arrays.stream(array).sum())).average().orElseThrow());
                }

                backProp(outputError);
                currentRow += batchSize;
            }
            costs.add(currentCosts.stream().mapToDouble(num -> num).average().orElseThrow());
        }
    }

    @Override
    public Object predict(Object X) {
        if (X.getClass() != Vector.class) {
            throw new IllegalArgumentException("Only Vectors are acceptable in MLP to predict a result");
        }
        return predict((Vector) X);
    }

    public Vector predict(Vector X) {
        if (!isCompiled) {
            throw new IllegalStateException("Model is not compiled! Compile model before using it");
        }
        return (Vector) feedForward(X);
    }

    private Tensor feedForward(Tensor input) {
        return feed(0, input);
    }

    private Tensor feed(int layerNumber, Tensor X) {
        Tensor activation = layers.get(layerNumber).getActivation(X);
        if (layerNumber == layers.size() - 1) {
            return activation;
        } else {
            return feed(layerNumber + 1, activation);
        }
    }

    private void backProp(Tensor error) {
        prop(layers.size() - 1, error);
    }

    private void prop(int layerNumber, Tensor error) {
        Layer currentLayer = layers.get(layerNumber);
        if (layerNumber != 0) {
            Layer previousLayer = layers.get(layerNumber - 1);
            currentLayer.updateParams(previousLayer.getLastActivation(), error, learningRate);
            Tensor newError = error.dot(currentLayer.getWeights().transpose()).elementwise(previousLayer.getActivationFunction().derivative(previousLayer.getLastImpulse()));
            prop(layerNumber - 1, newError);
        } else {
            currentLayer.updateParams(lastInput, error, learningRate);
        }
    }
}
