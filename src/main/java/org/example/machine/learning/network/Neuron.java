package org.example.machine.learning.network;

import org.example.algebra.Tensor;
import org.example.algebra.Vector;
import org.example.machine.learning.activation.ActivationFunction;

public class Neuron {
    private Vector weights;
    private Vector lastActivation;
    private Vector lastImpulse;
    private double bias;
    private ActivationFunction activationFunction;

    public Neuron(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public void compile(Shape inputShape, Initializer initializer){
        if(inputShape.getDimensions() != 1){
            throw new IllegalArgumentException("Neurons accept only one-dimensional data");
        }
        weights = (Vector) Tensor.build(new double[inputShape.getX()],false);
        bias = initializer.compute(weights);
    }

    public Vector computeOutput(Tensor X){
        if(weights.height() != X.width()) {
            throw new IllegalArgumentException("Dimension of input data doesn't match number of weights in neuron");
        }
        Vector neuronOutput = (Vector) X.dot(weights).addition(bias);
        lastImpulse = neuronOutput.isHorizontal() ? (Vector) neuronOutput.transpose() : neuronOutput; //this line and
        Vector neuronActivation = (Vector) activationFunction.call(lastImpulse);
        lastActivation = neuronActivation.isHorizontal() ? (Vector) neuronActivation.transpose() : neuronActivation; //this line are needed due to way of implementing dot product in Tensor class
        return lastActivation;                                                                                    //the reason why we need it is fact that activation vector must be not horizontal
    }                                                                                                                //and when we get vector with length 1 dot product makes it horizontal

    public void updateWeights(Vector X){
        if(weights.width() != X.height() && weights.width() != X.width()) {
            throw new IllegalArgumentException("Dimension of update data doesn't match number of weights in neuron");
        }

        weights = (Vector) weights.add(X);
    }

    public void updateBias(double x){
        bias += x;
    }

    public Vector getWeights(){
        return weights;
    }

    public Vector getLastActivation(){
        return lastActivation;
    }

    public Vector getLastImpulse() {
        return lastImpulse;
    }
}
