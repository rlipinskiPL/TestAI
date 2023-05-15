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
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.DoubleStream;

/**
 * This class implements multi layer perceptron model that accepts only MLPLayer and DropOutLayer as layers.
 * It is model that is based on neurons grouped in layers, all neurons in adjacent layers are connected to each other.
 * Only acceptable input for this model is two-dimensional tensors, but single data object is one-dimensional vector.
 */
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

    /**
     * This method trains the model based on the provided data.
     * It performs feed forward and back propagation to train model.
     *
     * @param input     training data
     * @param output    training labels
     * @param batchSize size of batch transferred to single training iteration
     * @param epochs    number of epochs
     * @param shuffle   if model should shuffle training data after each epoch
     * @see <a href="/equations.pdf>4 and 5 section in equations.pdf file</a>
     */
    public void fit(Tensor input, Tensor output, int batchSize, int epochs, boolean shuffle) {
        if (!isCompiled) {
            throw new IllegalStateException("Model is not compiled! Compile model before using it");
        } else if (batchSize < 0 || epochs < 0) {
            throw new IllegalArgumentException("Epochs and batchSize must be positive number");
        }

        Layer lastLayer = layers.get(layers.size() - 1);

        for (int i = 0; i < epochs; i++) {
            if (shuffle) {
                input = shuffle(input); //ToDo its wrong, so should fix it
            }

            int currentRow = 0;
            List<Double> currentCosts = new LinkedList<>();

            while (currentRow < input.height()) {
                Tensor X = input.cut(currentRow, currentRow + batchSize - 1);
                Tensor Y = output.cut(currentRow, currentRow + batchSize - 1);
                lastInput = X;

                Tensor netOutput = feedForward(X);

                Tensor outputError = computeError(  //EQUATION 5.1 (equations.pdf)
                        lossFunction.derivative(netOutput, Y),
                        lastLayer.getActivationFunction().derivative(lastLayer.getLastImpulse())
                );

                Tensor cost = lossFunction.call(netOutput, Y);
                if (cost.getClass() == Vector.class) {
                    currentCosts.add(((Vector) cost).stream().mapToDouble(d -> d).average().orElseThrow(NullPointerException::new));
                } else {
                    currentCosts.add(((Matrix) cost).stream().flatMapToDouble(array -> DoubleStream.of(Arrays.stream(array).sum())).average().orElseThrow(NullPointerException::new));
                }

                backProp(outputError);
                currentRow += batchSize;
            }
            Double meanCost = currentCosts.stream().mapToDouble(num -> num).average().orElseThrow(NullPointerException::new);
            costs.add(meanCost);
            System.out.println("Epoch " + (i + 1) + " loss: " + meanCost);
        }
    }

    @Override
    public Object predict(Object X) {
        if (X.getClass() != Vector.class) {
            throw new IllegalArgumentException("Only Vectors are acceptable in MLP to predict a result");
        }
        return predict((Vector) X);
    }

    /**
     * Predicts value basing on given data.
     *
     * @param X input data
     * @return predicted value
     */
    public Vector predict(Vector X) {
        if (!isCompiled) {
            throw new IllegalStateException("Model is not compiled! Compile model before using it");
        }
        return (Vector) feedForward(X);
    }

    /**
     * This method perform feed forward recursively.
     *
     * @param input input data
     * @return predicted value
     */
    private Tensor feedForward(Tensor input) {
        return feed(0, input);
    }

    /**
     * Recursive method propagating layers output forward.
     *
     * @param layerNumber number of layer to compute output
     * @param X           input data
     * @return output of given layer
     */
    private Tensor feed(int layerNumber, Tensor X) {
        Tensor activation = layers.get(layerNumber).computeOutput(X); //EQUATION 4.1 or 4.2 (equations.pdf)
        if (layerNumber == layers.size() - 1) {
            return activation;
        } else {
            return feed(layerNumber + 1, activation);
        }
    }

    /**
     * This method perform back propagation recursively.
     *
     * @param error last layer error
     */
    private void backProp(Tensor error) {
        prop(layers.size() - 1, error);
    }

    /**
     * Recursive method propagating error through all layers.
     *
     * @param layerNumber number of layer to update weights and compute new error
     * @param error       error for current layer
     */
    private void prop(int layerNumber, Tensor error) {
        Layer currentLayer = layers.get(layerNumber);
        if (layerNumber != 0) {
            Layer previousLayer = layers.get(layerNumber - 1);

            Tensor weightsDelta = previousLayer.getLastActivation().transpose()  //EQUATION 5.3 (equations.pdf)
                    .matmul(error)
                    .multiply(learningRate);
            Tensor biasDelta = error.multiply(learningRate);  //EQUATION 5.4 (equations.pdf)

            currentLayer.updateParams(
                    weightsDelta,
                    biasDelta
            );

            Tensor newError = computeError(  //EQUATION 5.2 (equations.pdf)
                    error.matmul(currentLayer.getWeights().transpose()),
                    previousLayer.getActivationFunction().derivative(previousLayer.getLastImpulse())
            );

            prop(layerNumber - 1, newError);
        } else {
            Tensor weightsDelta = lastInput.transpose()
                    .matmul(error)
                    .multiply(learningRate);
            Tensor biasDelta = error.multiply(learningRate);

            currentLayer.updateParams(
                    weightsDelta,
                    biasDelta
            );
        }
    }

    /**
     * This method computes error basing on two factors given as parameters.
     * First case is when two factors have the same shape, then result is hadamard product of them.
     * Second case is when two factors have the same width but second factor height is a product of height and width of first factor.
     * It means that second factor represents not vector of derivatives but jacobians combined into one matrix.
     *
     * @param first  first factor for computing error
     * @param second second factor for computing error
     * @return error
     * @see <a href="/equations.pdf>equation 5.2</a>
     */
    private Tensor computeError(Tensor first, Tensor second) {
        if (first.getShape().equals(second.getShape())) {
            return first.elementwise(second);
        } else if (first.height() * first.width() == second.height() && first.width() == second.width()) {
            if (first.isVector()) {
                return first.matmul(second);
            }

            Matrix firstMatrix = (Matrix) first;
            Matrix secondMatrix = (Matrix) second;
            Vector[] resultVectors = new Vector[first.height()];

            for (int i = 0; i < first.height(); i++) {
                resultVectors[i] = ((Vector) firstMatrix.getRow(i)
                        .matmul(secondMatrix.cut(
                                i * first.width(),
                                (i + 1) * first.width() - 1)
                        ));
            }
            return Tensor.makeMatrix(resultVectors);
        } else {
            throw new UnsupportedOperationException("Unsupported method of computing error in MLP");
        }
    }

    /**
     * NOT WORKING
     *
     * @param X input
     * @return shuffled input
     */
    private Tensor shuffle(Tensor X) {

        Matrix data = (Matrix) X;
        Integer[] indices = new Integer[data.height()];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = i;
        }
        Collections.shuffle(Arrays.asList(indices));

        Vector[] result = new Vector[data.height()];
        for (int i = 0; i < data.height(); i++) {
            result[i] = data.getRow(indices[i]);
        }

        return Tensor.makeMatrix(result);
    }
}
