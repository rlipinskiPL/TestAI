package org.example.algebra;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class Matrix extends Tensor {
    private double[][] data;

    public Matrix(List<List<Double>> values) {
        super(values.size(), values.get(0).size());
        if (values.size() == 1 || values.get(0).size() == 1) {
            throw new IllegalStateException("Matrix must be multidimensional, use Vector instead");
        }
        data = new double[values.size()][values.get(0).size()];
        for (int i = 0; i < values.size(); i++) {
            for (int j = 0; j < values.get(0).size(); j++) {
                data[i][j] = values.get(i).get(j);
            }
        }
    }

    public Matrix(double[][] values) {
        super(values.length, values[0].length);
        data = deepCopy(values);
    }

    @Override
    public double get(int i, int j) {
        return data[i][j];
    }

    @Override
    public void set(int i, int j, double value){
        data[i][j] = value;
    }

    @Override
    public double getAsScalar() {
        if (data.length == 1 && data[0].length == 1) {
            return data[0][0];
        } else {
            throw new IllegalStateException("Cannot parse to scalar because matrix is not 1x1");
        }
    }

    @Override
    public Tensor transpose() {
        //ToDo find better algorithm or optimize it by java functionalities
        double[][] result = new double[width][height];
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                result[j][i] = data[i][j];
            }
        }
        return Tensor.build(result);
    }

    @Override
    public Tensor multiply(double value) {
        double[][] result = new double[height][width];
        for (int i = 0; i < data.length; i++) {
            result[i] = Arrays.stream(data[i]).map(o -> value * o).toArray();
        }
        return Tensor.build(result);
    }

    @Override
    public Tensor addition(double value) {
        double[][] result = new double[height][width];
        for (int i = 0; i < data.length; i++) {
            result[i] = Arrays.stream(data[i]).map(o -> value + o).toArray();
        }
        return Tensor.build(result);
    }

    @Override
    public Tensor power(double value) {
        double[][] result = new double[height][width];
        for (int i = 0; i < data.length; i++) {
            result[i] = Arrays.stream(data[i]).map(o -> java.lang.Math.pow(o,value)).toArray();
        }
        return Tensor.build(result);
    }

    @Override
    public Tensor cut(int start, int end) {
        if (start > end) {
            throw new IllegalArgumentException("Start must be smaller value than end");
        } else if (start >= data.length) {
            throw new IllegalArgumentException("Start must be smaller than height of matrix");
        }

        double[][] result = end < this.height ? new double[end - start + 1][data[0].length] : new double[this.height - start + 1][data[0].length];
        for (int i = start, k = 0; i <= end; i++, k++) {
            if(i == data.length)
                break;

            result[k] = data[i];
        }
        return Tensor.build(result);
    }

    @Override
    public java.lang.Object clone() {
        return Tensor.build(deepCopy(data));
    }

    @Override
    public void print() {
        for (double[] datum : data) {
            StringBuilder builder = new StringBuilder();
            builder.append("[");
            for (int j = 0; j < data[0].length; j++) {
                builder.append(datum[j]);
                builder.append("|");
            }
            builder.deleteCharAt(builder.length() - 1);
            builder.append("]");
            System.out.println(builder);
        }
    }

    @Override
    public Stream<double[]> stream() {
        return Arrays.stream(data);
    }

    public Vector getRow(int n) {
        return (Vector) Tensor.build(data[n].clone(), true);
    }

    public Vector dropRow(int n) {
        double[][] result = new double[height - 1][width];
        Tensor toReturn = null;
        for (int i = 0, k = 0; i < data.length; i++) {
            if (i != n) {
                result[k] = data[i];
            } else {
                k--;
                toReturn = Tensor.build(data[i].clone(), true);
            }
            i++;
            k++;
        }
        data = result;
        height--;
        return (Vector) toReturn;
    }

    public Vector getColumn(int n) {
        double[] result = new double[height];
        for (int i = 0; i < height; i++) {
            result[i] = data[i][n];
        }
        return (Vector) Tensor.build(result, false);
    }

    public Vector dropColumn(int n) {
        double[] toReturn = new double[height];
        double[][] result = new double[height][width - 1];

        for (int i = 0; i < height; i++) {
            for (int j = 0, k = 0; j < width; j++) {
                if (j != n) {
                    result[i][k] = data[i][j];
                } else {
                    k--;
                    toReturn[i] = data[i][j];
                }
                j++;
                k++;
            }
        }
        data = result;
        width--;
        return (Vector) Tensor.build(toReturn, false);
    }

    private static double[][] deepCopy(double[][] input) {
        if (input == null)
            return null;
        double[][] result = new double[input.length][];
        for (int r = 0; r < input.length; r++) {
            result[r] = input[r].clone();
        }
        return result;
    }
}
