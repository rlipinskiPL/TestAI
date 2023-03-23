package ai.test.machine.learning.layers;

import ai.test.algebra.Matrix;
import ai.test.algebra.Shape;
import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;
import ai.test.machine.learning.activations.ActivationFunction;
import ai.test.machine.learning.activations.JointValuesActivation;
import ai.test.machine.learning.initializers.Initializer;
import ai.test.machine.learning.initializers.RandomNormal;
import ai.test.machine.learning.layers.neurons.Neuron;
import ai.test.machine.learning.tools.Regularizer;

import java.util.ArrayList;
import java.util.List;

public class MLPLayer implements Layer {
    private final List<Neuron> neurons = new ArrayList<>();
    private final ActivationFunction activationFunction;
    private Regularizer regularizer = Regularizer.NON;
    private Initializer initializer = new RandomNormal();

    public MLPLayer(int units, ActivationFunction activationFunction, Regularizer regularizer, Initializer initializer) {
        for (int i = 0; i < units; i++) {
            neurons.add(new Neuron(activationFunction));
        }
        this.activationFunction = activationFunction;
        this.regularizer = regularizer;
        this.initializer = initializer;
    }

    public MLPLayer(int units, ActivationFunction activationFunction, Regularizer regularizer) {
        for (int i = 0; i < units; i++) {
            neurons.add(new Neuron(activationFunction));
        }
        this.activationFunction = activationFunction;
        this.regularizer = regularizer;
    }

    public MLPLayer(int units, ActivationFunction activationFunction, Initializer initializer) {
        for (int i = 0; i < units; i++) {
            neurons.add(new Neuron(activationFunction));
        }
        this.activationFunction = activationFunction;
        this.initializer = initializer;
    }

    public MLPLayer(int units, ActivationFunction activationFunction) {
        for (int i = 0; i < units; i++) {
            neurons.add(new Neuron(activationFunction));
        }
        this.activationFunction = activationFunction;
    }

    @Override
    public void updateParams(Tensor inputs, Tensor error, double learningRate) {
        Tensor updateWeights = inputs.transpose().dot(error).multiply(-learningRate);
        Tensor updateBiases = error.multiply(-learningRate);
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            if (updateWeights.isVector()) {
                neuron.updateWeights((Vector) updateWeights);
            } else {
                neuron.updateWeights(((Matrix) updateWeights).getColumn(i));
            }

            if (error.isVector()) {
                neuron.updateBias(((Vector) updateBiases).get(i));
            } else {
                neuron.updateBias(((Matrix) updateBiases).getColumn(i).stream().mapToDouble(d -> d).sum());
            }
        }
    }

    @Override
    public Tensor getActivation(Tensor input) {
        Vector[] activations = new Vector[neurons.size()];
        for (int i = 0; i < neurons.size(); i++) {
            activations[i] = neurons.get(i).computeOutput(input);
        }

        Tensor toReturn = Tensor.makeMatrix(activations);
        if (activationFunction instanceof JointValuesActivation) {
            toReturn = ((JointValuesActivation) activationFunction).callJointly(toReturn);
        }

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

    @Override
    public void compile(Shape inputShape) {
        if (inputShape.getDimensions() != 1) {
            throw new IllegalArgumentException("Neurons accept only one-dimensional data");
        }
        neurons.forEach(neuron -> neuron.compile(inputShape, initializer));
    }

    @Override
    public Shape getShape() {
        return new Shape(neurons.size());
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }
}