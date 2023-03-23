package org.example.machine.learning.activation;

import org.example.algebra.Tensor;

public interface ActivationFunction {
    default Tensor call(Tensor X){
        double[][] result = new double[X.height()][X.width()];
        for(int i=0;i<X.height();i++){
            for(int j=0;j<X.width();j++){
                result[i][j] = call(X.get(i,j));
            }
        }
        return Tensor.build(result);
    }

    default Tensor derivative(Tensor X){
        double[][] result = new double[X.height()][X.width()];
        for(int i=0;i<X.height();i++){
            for(int j=0;j<X.width();j++){
                result[i][j] = derivative(X.get(i,j));
            }
        }
        return Tensor.build(result);
    }

    double call(double x);

    double derivative(double x);
}
