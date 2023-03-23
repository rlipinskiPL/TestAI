package org.example.machine.learning.network;

import org.example.algebra.Tensor;

public interface Initializer {
    double compute(Tensor tensor);
}
