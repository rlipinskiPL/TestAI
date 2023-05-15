package ai.test.machine.learning.initializers;

import ai.test.algebra.Tensor;

/**
 * This interface is responsible for initializing all hiperparameters in single neuron.
 * Neuron in this context means single computing unit in AI model layer.
 */
public interface Initializer {

    /**
     * This method gets tensor with hiperparamters of computing unit
     * and initialize all values in this tensor in accordance to specific algorithm
     *
     * @param tensor tensor with neuron hiperparameters
     * @return new bias value
     */
    double call(Tensor tensor);
}
