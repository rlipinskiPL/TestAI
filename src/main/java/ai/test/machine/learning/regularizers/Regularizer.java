package ai.test.machine.learning.regularizers;

import ai.test.algebra.Tensor;

/**
 * This interface is intended for all algorithms that prevents from overfitting
 * by adding some value to update weights vector basing on weights values.
 * It specifies only one method - computeDerivative, since only this method is necessary to perform anti overfitting action.
 */
public interface Regularizer {

    /**
     * This method computes derivative of regularizer function needed for adding it to update weights vector.
     *
     * @param weights weights of model layer
     * @return derivative of regularizer function
     */
    Tensor computeDerivative(Tensor weights);
}
