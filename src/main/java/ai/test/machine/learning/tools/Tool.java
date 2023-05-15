package ai.test.machine.learning.tools;

import ai.test.algebra.Tensor;


public interface Tool {

    void fit(Tensor X);

    Tensor transform(Tensor X);
}
