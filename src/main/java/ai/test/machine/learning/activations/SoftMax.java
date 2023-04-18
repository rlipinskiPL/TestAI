package ai.test.machine.learning.activations;

import ai.test.algebra.Matrix;
import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;

import java.util.Arrays;

//ToDo this implementation is wrong, need to fix it
public class SoftMax extends JointValuesActivation {

    @Override
    public Tensor callJointly(Tensor X) {
        X = (Tensor) X.clone();
        double[] max = new double[X.height()];
        for (int i = 0; i < max.length; i++) {
            max[i] = -Double.MAX_VALUE;
        }

        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                if (X.get(i, j) > max[i]) {
                    max[i] = X.get(i, j);
                }
            }
        }

        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                X.set(i, j, X.get(i, j) - max[i]);
            }
        }

        double[][] nominators = new double[X.height()][X.width()];
        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                nominators[i][j] = Math.exp(X.get(i, j));
            }
        }

        double[] denominators = new double[X.height()];
        for (int i = 0; i < X.height(); i++) {
            denominators[i] = Arrays.stream(nominators[i]).sum();
        }

        double[][] result = new double[X.height()][X.width()];
        for (int i = 0; i < X.height(); i++) {
            for (int j = 0; j < X.width(); j++) {
                result[i][j] = nominators[i][j] / denominators[i];
            }
        }
        
        return Tensor.build(result);
    }

    @Override
    public Tensor derivative(Tensor X) {
        int width = X.width();
        double[][] result = new double[width*X.height()][width];
        for(int rowNr=0;rowNr<X.height();rowNr++){
            Vector row;
            if(X.getClass()== Matrix.class){
                row = ((Matrix)X).getRow(rowNr);
            }else{
                row = (Vector) X;
            }

            Vector softmax = (Vector) callJointly(row);
            for(int i=0;i<width;i++){
                for(int j=0;j<width;j++){
                    if(i == j){
                        result[rowNr*width+i][j] = softmax.get(i)*(1- softmax.get(i));
                    }else{
                        result[rowNr*width+i][j] = -softmax.get(i)* softmax.get(j);
                    }
                }
            }
        }
        return Tensor.build(result);
    }
}
