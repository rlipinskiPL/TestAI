package org.example.machine.learning.network;

import org.example.algebra.Tensor;
import org.example.machine.learning.activation.ActivationFunction;
import org.example.machine.learning.loss.LossFunction;

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
