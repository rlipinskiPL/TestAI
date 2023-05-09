package ai.test.machine.learning.layers;

import ai.test.algebra.Shape;
import ai.test.algebra.Tensor;
import ai.test.machine.learning.activations.ActivationFunction;

public interface Layer {

    void updateParams(Tensor inputs, Tensor error, double learningRate);

    void compile(Shape inputShape);

    Tensor getActivation(Tensor input);

    Tensor getLastActivation();

    Tensor getLastImpulse();

    Tensor getWeights();

    Shape getShape();

    ActivationFunction getActivationFunction();
}
