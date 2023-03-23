package org.example.machine.learning;

import org.example.algebra.Matrix;
import org.example.algebra.Vector;
import org.example.algebra.Tensor;

import java.util.ArrayList;
import java.util.List;

public class Standardization {
    private List<Double> mu;
    private List<Double> sigma;

    public void fit(Matrix X) {
        mu = new ArrayList<>(X.width());
        sigma = new ArrayList<>(X.width());

        for (int i = 0; i < X.width(); i++) {
            Vector column = (Vector) X.getColumn(i);
            mu.add(column.stream().mapToDouble(d->d).sum() / column.height());
            sigma.add(Math.sqrt(column.stream().mapToDouble(d->d).map(o -> Math.pow(o - mu.get(mu.size() - 1), 2)).sum() / column.height()));
        }
    }

    public Tensor transform(Matrix X){
        if(X.width() != mu.size()){
            throw new IllegalArgumentException("Width of matrix is not equal to matrix used to fit");
        }

        double[][] result = new double[X.height()][X.width()];

        for(int i=0;i<X.width();i++){
            for(int j=0;j<X.height();j++){
                result[j][i] = (X.get(j,i)-mu.get(i))/sigma.get(i);
            }
        }

        return Tensor.build(result);
    }
}
