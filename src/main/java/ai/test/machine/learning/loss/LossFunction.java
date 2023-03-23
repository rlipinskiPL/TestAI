package ai.test.machine.learning.loss;

import ai.test.algebra.Tensor;

public interface LossFunction {
    default Tensor call(Tensor X, Tensor Y) {
        double[][] result = new double[X.height()][X.width()];
        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                result[i][j] = call(X.get(i, j), Y.get(i, j));
            }
        }
        return Tensor.build(result);
    }

    default Tensor derivative(Tensor X, Tensor Y) {
        double[][] result = new double[X.height()][X.width()];
        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                result[i][j] = derivative(X.get(i, j), Y.get(i, j));
            }
        }
        return Tensor.build(result);
    }

    double call(double x, double y);

    double derivative(double x, double y);
}
