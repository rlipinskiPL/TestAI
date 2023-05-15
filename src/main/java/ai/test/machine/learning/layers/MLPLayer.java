package ai.test.machine.learning.layers;

import ai.test.algebra.Matrix;
import ai.test.algebra.Shape;
import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;
import ai.test.machine.learning.activations.ActivationFunction;
import ai.test.machine.learning.initializers.Initializer;
import ai.test.machine.learning.initializers.RandomNormal;
import ai.test.machine.learning.layers.neurons.Neuron;
import ai.test.machine.learning.regularizers.Regularizer;
import lombok.Getter;

import java.util.ArrayList;
import java.util.List;

/**
 * Layer interface implementation according to multi layer perceptron where all computing units are connected to each other.
 */
public class MLPLayer implements Layer {

    private final List<Neuron> neurons = new ArrayList<>();

    private final ActivationFunction activationFunction;

    private Regularizer regularizer = null;

    private Initializer initializer = new RandomNormal();

    @Getter
    private final Shape shape;

    public MLPLayer(
            int units,
            ActivationFunction activationFunction,
            Regularizer regularizer,
            Initializer initializer
    ) {
        for (int i = 0; i < units; i++) {
            neurons.add(new Neuron(activationFunction));
        }
        this.activationFunction = activationFunction;
        this.regularizer = regularizer;
        this.initializer = initializer;
        this.shape = new Shape(units);
    }

    public MLPLayer(
            int units,
            ActivationFunction activationFunction,
            Regularizer regularizer
    ) {
        for (int i = 0; i < units; i++) {
            neurons.add(new Neuron(activationFunction));
        }
        this.activationFunction = activationFunction;
        this.regularizer = regularizer;
        this.shape = new Shape(units);
    }

    public MLPLayer(int units,
                    ActivationFunction activationFunction,
                    Initializer initializer) {
        for (int i = 0; i < units; i++) {
            neurons.add(new Neuron(activationFunction));
        }
        this.activationFunction = activationFunction;
        this.initializer = initializer;
        this.shape = new Shape(units);
    }

    public MLPLayer(int units, ActivationFunction activationFunction) {
        for (int i = 0; i < units; i++) {
            neurons.add(new Neuron(activationFunction));
        }
        this.activationFunction = activationFunction;
        this.shape = new Shape(units);
    }

    /**
     * {@inheritDoc}
     *
     * @param updateWeights tensor of update weights vectors
     * @param biasesDelta   tensor of update bias scalars
     * @implNote This implementation supports regularizer, so the final result of method could be different from
     * standard updating hiperparameters algorithm
     */
    @Override
    public void updateParams(Tensor updateWeights, Tensor biasesDelta) {
        if (regularizer != null) {
            updateWeights = updateWeights.add(regularizer.computeDerivative(getWeights())); //applying regularizer value
        }

        for (int i = 0; i < neurons.size(); i++) { //passing the appropriate update vector to appropriate neuron
            Neuron neuron = neurons.get(i);
            if (updateWeights.isVector()) {
                neuron.updateWeights((Vector) updateWeights);
            } else {
                neuron.updateWeights(((Matrix) updateWeights).getColumn(i));
            }

            if (biasesDelta.isVector()) {
                neuron.updateBias(((Vector) biasesDelta).get(i));
            } else {
                neuron.updateBias(((Matrix) biasesDelta).getColumn(i).stream().mapToDouble(d -> d).sum());
            }
        }
    }

    /**
     * {@inheritDoc}
     *
     * @param input input data
     * @return
     * @implNote This implementation collect all neurons activations that are computed according to equation number 3.2 in equations.pdf file.
     * @see <a href="equations.pdf">3.2 equation</a>
     */
    @Override
    public Tensor computeOutput(Tensor input) {
        Vector[] activations = new Vector[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            activations[i] = neurons.get(i).computeOutput(input); //combining neurons outputs into one tensor
        }
        Tensor toReturn = Tensor.makeMatrix(activations);

        return toReturn;
    }

    @Override
    public Tensor getLastActivation() {
        Vector[] activations = new Vector[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            activations[i] = neurons.get(i).getLastActivation();
        }
        return Tensor.makeMatrix(activations);
    }

    @Override
    public Tensor getLastImpulse() {
        Vector[] impulses = new Vector[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            impulses[i] = neurons.get(i).getLastImpulse();
        }
        return Tensor.makeMatrix(impulses);
    }

    @Override
    public Tensor getWeights() {
        Vector[] weights = new Vector[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            weights[i] = neurons.get(i).getWeights();
        }
        return Tensor.makeMatrix(weights);
    }

    /**
     * {@inheritDoc}
     *
     * @param inputShape shape of the input data that will be delivered to the layer
     * @throws IllegalArgumentException when input shape is not one-dimensional
     */
    @Override
    public void compile(Shape inputShape) {
        if (inputShape.getDimensions() != 1) {
            throw new IllegalArgumentException("Neurons accept only one-dimensional data");
        }
        neurons.forEach(neuron -> neuron.compile(inputShape, initializer));
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}
