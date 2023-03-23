package org.example.machine.learning.activation;

import org.example.algebra.Tensor;

import java.util.Arrays;

public class SoftMax extends JointValuesActivation{

    @Override
    public Tensor callJointly(Tensor X) { //ToDo too many for loops
        X = (Tensor) X.clone();
        double[] max = new double[X.height()];
        for(int i=0;i<max.length;i++){
            max[i] = -Double.MAX_VALUE;
        }

        for(int i=0;i<X.height();i++){
            for(int j=0;j<X.width();j++){
                if(X.get(i,j)>max[i]){
                    max[i] = X.get(i,j);
                }
            }
        }

        for(int i=0;i<X.height();i++){
            for(int j=0;j<X.width();j++){
                X.set(i,j,X.get(i,j)-max[i]);
            }
        }

        double[][] nominators = new double[X.height()][X.width()];
        for(int i=0;i<X.height();i++){
            for(int j=0;j<X.width();j++){
                nominators[i][j] = Math.exp(X.get(i,j));
            }
        }

        double[] denominators = new double[X.height()];
        for(int i=0;i<X.height();i++){
            denominators[i]= Arrays.stream(nominators[i]).sum();
        }

        double[][] result = new double[X.height()][X.width()];
        for(int i=0;i<X.height();i++){
            for(int j=0;j<X.width();j++){
                result[i][j] = nominators[i][j]/denominators[i];
            }
        }

        return Tensor.build(result);
    }

    @Override
    public Tensor derivative(Tensor X) {
        Tensor resultOfSoftMax = this.call(X);
        double[][] result = new double[X.height()][X.width()];
        for(int i=0;i<X.height();i++){
            for(int j=0;j<X.width();j++){
                double value = resultOfSoftMax.get(i,j);
                result[i][j] = value*(1-value);
            }
        }
        return Tensor.build(result);
    }
}
