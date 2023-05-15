package ai.test.machine.learning.layers.neurons;

import ai.test.algebra.Shape;
import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;
import ai.test.machine.learning.activations.ActivationFunction;
import ai.test.machine.learning.initializers.Initializer;
import lombok.Getter;

/**
 * Single computing unit for MLP model.
 */
public class Neuron {

    /**
     * weights of neuron
     */
    @Getter
    private Vector weights;

    /**
     * last activation of neuron
     */
    @Getter
    private Vector lastActivation;

    /**
     * last impulse of neuron
     */
    @Getter
    private Vector lastImpulse;

    private double bias;

    private final ActivationFunction activationFunction;

    public Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * This method initialize number of weights and their values.
     *
     * @param inputShape  shape of the input data that will be delivered to the neuron
     * @param initializer instance of initializer that will initialize weights values of this neuron
     */
    public void compile(Shape inputShape, Initializer initializer) {
        if (inputShape.getDimensions() != 1) {
            throw new IllegalArgumentException("Neurons accept only one-dimensional data");
        }

        weights = (Vector) Tensor.build(new double[inputShape.getX()], false);
        bias = initializer.call(weights);
    }

    /**
     * Calculate activation of neuron according to equation number 3.2 in equations.pdf file.
     *
     * @param X input tensor
     * @return activation of neuron
     * @throws IllegalArgumentException when number of inputs doesn't match number of weights
     * @see <a href="equations.pdf">equation 3.2</a>
     */
    public Vector computeOutput(Tensor X) {
        if (weights.height() != X.width()) {
            throw new IllegalArgumentException("Dimension of input data doesn't match number of weights in neuron");
        }

        Vector neuronOutput = (Vector) X.matmul(weights).addition(bias); //EQUATION 3.1 (equations.pdf)
        lastImpulse = neuronOutput.isHorizontal() ? (Vector) neuronOutput.transpose() : neuronOutput; //this line and
        Vector neuronActivation = (Vector) activationFunction.call(lastImpulse); //EQUATION 3.2 (equations.pdf)
        lastActivation = neuronActivation.isHorizontal() ? (Vector) neuronActivation.transpose() : neuronActivation; //this line are needed due to way of implementing dot product in Tensor class
        return lastActivation;                                                                                    //the reason why we need it is fact that activation vector must be vertical
    }                                                                                                                //and when we get vector with length 1 dot product makes it horizontal

    /**
     * Update weights according to equation number 3.3 in equations.pdf file.
     *
     * @param X update weights vector
     * @see <a href="equations.pdf">equation 3.3</a>
     */
    public void updateWeights(Vector X) {
        if (weights.width() != X.height() && weights.width() != X.width()) {
            throw new IllegalArgumentException("Dimension of update data doesn't match number of weights in neuron");
        }

        weights = (Vector) weights.sub(X); //EQUATION 3.3 (equations.pdf)
    }

    /**
     * Update bias according to equation number 3.4 in equations.pdf file.
     *
     * @param x update bias scalar
     * @see <a href="equations.pdf">equation 3.4</a>
     */
    public void updateBias(double x) {
        bias -= x; //EQUATION 3.4 (equations.pdf)
    }
}
