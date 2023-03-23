package org.example.machine.learning.network;

import org.example.algebra.Tensor;
import org.example.machine.learning.activation.ActivationFunction;

public class DropOutLayer implements Layer{

    private final double rate;

    public DropOutLayer(double rate) {
        this.rate = rate;
    }

    @Override
    public void updateParams(Tensor inputs, Tensor error, double learningRate) {
        return;
    }

    @Override
    public Tensor getActivation(Tensor input) {
        return null;
    }

    @Override
    public Tensor getWeights() {
        return null;
    }

    @Override
    public void compile(Shape inputShape) {

    }

    @Override
    public Shape getShape() {
        return null;
    }

    @Override
    public Tensor getLastActivation() {
        return null;
    }

    @Override
    public Tensor getLastImpulse() {
        return null;
    }

    @Override
    public ActivationFunction getActivationFunction() {
        return null;
    }
}
