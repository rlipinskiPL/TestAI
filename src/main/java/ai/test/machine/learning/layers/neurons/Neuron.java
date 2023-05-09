package ai.test.machine.learning.layers.neurons;

import ai.test.algebra.Shape;
import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;
import ai.test.machine.learning.activations.ActivationFunction;
import ai.test.machine.learning.initializers.Initializer;
import lombok.Getter;

public class Neuron {

    @Getter
    private Vector weights;

    @Getter
    private Vector lastActivation;

    @Getter
    private Vector lastImpulse;

    private double bias;

    private final ActivationFunction activationFunction;

    public Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public void compile(Shape inputShape, Initializer initializer) {
        if (inputShape.getDimensions() != 1) {
            throw new IllegalArgumentException("Neurons accept only one-dimensional data");
        }

        weights = (Vector) Tensor.build(new double[inputShape.getX()], false);
        bias = initializer.compute(weights);
    }

    public Vector computeOutput(Tensor X) {
        if (weights.height() != X.width()) {
            throw new IllegalArgumentException("Dimension of input data doesn't match number of weights in neuron");
        }

        Vector neuronOutput = (Vector) X.dot(weights).addition(bias);
        lastImpulse = neuronOutput.isHorizontal() ? (Vector) neuronOutput.transpose() : neuronOutput; //this line and
        Vector neuronActivation = (Vector) activationFunction.call(lastImpulse);
        lastActivation = neuronActivation.isHorizontal() ? (Vector) neuronActivation.transpose() : neuronActivation; //this line are needed due to way of implementing dot product in Tensor class
        return lastActivation;                                                                                    //the reason why we need it is fact that activation vector must be vertical
    }                                                                                                                //and when we get vector with length 1 dot product makes it horizontal

    public void updateWeights(Vector X) {
        if (weights.width() != X.height() && weights.width() != X.width()) {
            throw new IllegalArgumentException("Dimension of update data doesn't match number of weights in neuron");
        }

        weights = (Vector) weights.add(X);
    }

    public void updateBias(double x) {
        bias += x;
    }
}
