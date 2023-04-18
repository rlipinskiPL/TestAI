package ai.test.machine.learning.regularizers;

import ai.test.algebra.Tensor;

public class L2 implements Regularizer{
    private final double alpha;

    public L2(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public Tensor computeDerivative(Tensor weights) {
        double[][] result = new double[weights.height()][weights.width()];

        for(int i=0;i< weights.height();i++){
            for(int j=0;j< weights.width();j++){
                result[i][j]=alpha* weights.get(i,j);
            }
        }
        return Tensor.build(result);
    }
}
