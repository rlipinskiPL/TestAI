package ai.test.machine.learning.tools;

import ai.test.algebra.Matrix;
import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;

import java.util.ArrayList;
import java.util.List;

public class Standardization implements Tool {

    private List<Double> mu;

    private List<Double> sigma;

    @Override
    public void fit(Tensor X) {
        if (X.isVector()) {
            throw new IllegalArgumentException("Only Matrices are acceptable in Standardization");
        }

        Matrix Xmatrix = (Matrix) X;
        mu = new ArrayList<>(X.width());
        sigma = new ArrayList<>(X.width());

        for (int i = 0; i < Xmatrix.width(); i++) {
            Vector column = Xmatrix.getColumn(i);
            mu.add(column.stream().mapToDouble(d -> d).sum() / column.height());
            sigma.add(Math.sqrt(column.stream().mapToDouble(d -> d).map(o -> Math.pow(o - mu.get(mu.size() - 1), 2)).sum() / column.height()));
        }
    }

    @Override
    public Tensor transform(Tensor X) {
        if (X.width() != mu.size()) {
            throw new IllegalArgumentException("Width of matrix is not equal to matrix used to fit");
        }

        double[][] result = new double[X.height()][X.width()];

        for (int i = 0; i < X.width(); i++) {
            for (int j = 0; j < X.height(); j++) {
                result[j][i] = (X.get(j, i) - mu.get(i)) / sigma.get(i);
            }
        }

        return Tensor.build(result);
    }
}
