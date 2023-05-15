package ai.test.machine.learning.layers;

import ai.test.algebra.Shape;
import ai.test.algebra.Tensor;
import ai.test.machine.learning.activations.ActivationFunction;

/**
 * Single layer in deep learning models.
 */
public interface Layer {

    /**
     * Update hiperparameters of all neurons in layer
     *
     * @param weightsDelta tensor of update weights vectors
     * @param biasesDelta  tensor of update bias scalars
     */
    void updateParams(Tensor weightsDelta, Tensor biasesDelta);

    /**
     * This method initialize number of expected number of inputs and compile all neurons in layer.
     *
     * @param inputShape shape of the input data that will be delivered to the layer
     */
    void compile(Shape inputShape);

    /**
     * Return activation of all neurons as tensor.
     *
     * @param input input data
     * @return output
     */
    Tensor computeOutput(Tensor input);

    /**
     * @return last activation of all neurons in layer
     */
    Tensor getLastActivation();

    /**
     * @return last impulse of all neurons in layer
     */
    Tensor getLastImpulse();

    /**
     * @return weights of all neurons in layer
     */
    Tensor getWeights();

    /**
     * @return shape of layer output
     */
    Shape getShape();

    /**
     * @return activation function in layer
     */
    ActivationFunction getActivationFunction();
}
