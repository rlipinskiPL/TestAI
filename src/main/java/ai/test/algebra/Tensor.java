package ai.test.algebra;

import lombok.Getter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Stream;

//ToDo in subclasses protection against going outside the array
public abstract class Tensor implements Cloneable {

    protected int width;

    protected int height;

    @Getter
    protected Shape shape;

    protected Tensor(int height, int width) {
        if(height<0 || width < 0){
            throw new IllegalArgumentException("Dimensions of the tensor cannot be negative numbers");
        }
        this.width = width;
        this.height = height;
        this.shape = new Shape(height, width);
    }

    public abstract Tensor transpose();

    public abstract Tensor multiply(double value);

    public abstract Tensor addition(double value);

    public abstract Tensor power(double value);

    public abstract double get(int i, int j);

    public abstract void set(int i, int j, double value);

    public abstract double getAsScalar();

    public abstract Tensor cut(int start, int end);

    public abstract void print();

    public abstract Stream<?> stream();

    @Override
    public abstract java.lang.Object clone();

    public Tensor dot(Tensor tensor) {
        //ToDo implement better algorithm for matrices multiplication, this one sucks
        if (this.width != tensor.height) {
            throw new ArithmeticException("Incorrect dimensions of matrices (multiplication impossible)");
        }
        double[][] result = new double[this.height][tensor.width()];

        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[row].length; col++) {
                result[row][col] = multiplyMatricesCell(this, tensor, row, col);
            }
        }

        return Tensor.build(result);
    }

    public Tensor elementwise(Tensor tensor) {
        if (!this.shape.equals(tensor.shape)) {
            throw new ArithmeticException("Incorrect dimensions of matrices (elementwise multiplication impossible)");
        }
        double[][] result = new double[this.height][this.width];

        for (int row = 0; row < result.length; row++) {
            for (int col = 0; col < result[row].length; col++) {
                result[row][col] = this.get(row, col) * tensor.get(row, col);
            }
        }

        return Tensor.build(result);
    }

    public Tensor add(Tensor tensor) {
        if (!this.shape.equals(tensor.shape)) {
            throw new ArithmeticException("Incorrect dimensions of matrices (addition impossible)");
        }

        double[][] result = new double[height][width];

        for (int i = 0; i < this.height; i++) {
            for (int j = 0; j < this.width; j++) {
                result[i][j] = this.get(i, j) + tensor.get(i, j);
            }
        }
        return Tensor.build(result);
    }

    public Tensor sub(Tensor tensor) {
        if (!this.shape.equals(tensor.shape)) {
            throw new ArithmeticException("Incorrect dimensions of matrices (addition impossible)");
        }

        double[][] result = new double[height][width];

        for (int i = 0; i < this.height; i++) {
            for (int j = 0; j < this.width; j++) {
                result[i][j] = this.get(i, j) - tensor.get(i, j);
            }
        }
        return Tensor.build(result);
    }

    public boolean isVector() {
        return width == 1 || height == 1;
    }

    public int height() {
        return height;
    }

    public int width() {
        return width;
    }

    @Override
    public int hashCode() {
        int result = 7;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int c = (int) Double.doubleToLongBits(get(i, j));
                result = result * 37 + c;
            }
        }
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj.getClass() != this.getClass()) {
            return false;
        }
        Tensor tensor = (Tensor) obj;
        if (!tensor.shape.equals(this.shape)) {
            return false;
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (tensor.get(i, j) != this.get(i, j)) {
                    return false;
                }
            }
        }
        return true;
    }

    public boolean equalsWithPrecision(Object obj, double precision) {
        if (obj.getClass() != this.getClass()) {
            return false;
        }
        Tensor tensor = (Tensor) obj;
        if (!tensor.shape.equals(this.shape)) {
            return false;
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                double difference = Math.abs(this.get(i, j) - tensor.get(i, j));
                if (difference > precision) {
                    return false;
                }
            }
        }
        return true;
    }

    public static Tensor build(List<List<Double>> values) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("Empty list is not permitted here");
        }

        if (values.size() > 1 && values.get(0).size() > 1) {
            return new Matrix(values);
        } else {
            if (values.size() == 1) {
                return new Vector(values.get(0), true);
            } else {
                List<Double> tmp = new ArrayList<>();
                for (List<Double> value : values) {
                    tmp.add(value.get(0));
                }
                return new Vector(tmp, false);
            }
        }
    }

    public static Tensor build(double[][] values) {
        if (values.length == 0) {
            throw new IllegalArgumentException("Empty array is not permitted here");
        }

        if (values.length > 1 && values[0].length > 1) {
            return new Matrix(values);
        } else {
            if (values.length == 1) {
                return new Vector(values[0], true);
            } else {
                List<Double> tmp = new ArrayList<>();
                for (int i = 0; i < values.length; i++) {
                    tmp.add(values[i][0]);
                }
                return new Vector(tmp, false);
            }
        }
    }

    public static Tensor build(double[] values, boolean horizontal) {
        if (values.length == 0) {
            throw new IllegalArgumentException("Empty array is not permitted here");
        }
        return new Vector(values, horizontal);
    }

    public static Tensor build(List<Double> values, boolean horizontal) {
        if (values.size() == 0) {
            throw new IllegalArgumentException("Empty list is not permitted here");
        }
        return new Vector(values, horizontal);
    }

    public static Tensor build(boolean isHorizontal, double... values) {
        return Tensor.build(values, isHorizontal);
    }


    public static Tensor eye(int dimension) {
        double[][] result = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                result[i][j] = i == j ? 1 : 0;
            }
        }
        return Tensor.build(result);
    }

    public static Tensor readFromCsv(String filePath) {
        List<List<String>> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                records.add(Arrays.asList(values));
            }
        } catch (IOException e) {
            throw new RuntimeException(e.getMessage());
        }

        Map<String, Double> mapper = new HashMap<>();
        List<List<Double>> values = new ArrayList<>();

        for (var row : records) {
            values.add(new ArrayList<>());
            for (var cell : row) {
                try {
                    values.get(values.size() - 1).add(Double.parseDouble(cell));
                } catch (NumberFormatException e) {
                    if (mapper.containsKey(cell)) {
                        values.get(values.size() - 1).add(mapper.get(cell));
                    } else {
                        mapper.put(cell, (double) mapper.size());
                        values.get(values.size() - 1).add(mapper.get(cell));
                    }
                }
            }
        }
        return Tensor.build(values);
    }

    public static Tensor makeMatrix(Vector... vectors) {
        if (vectors.length == 1) {
            return vectors[0];
        } else if (vectors.length == 0) {
            throw new IllegalArgumentException("More than 0 vectors is needed to build a tensor");
        } else {
            for (int i = 0; i < vectors.length; i++) {
                if (!vectors[0].getShape().equals(vectors[i].getShape())) {
                    throw new IllegalArgumentException("All of passed vectors must have the same shape");
                }
            }
            double[][] result;
            if (vectors[0].isHorizontal()) {
                result = new double[vectors.length][vectors[0].width];
                for (int i = 0; i < vectors.length; i++) {
                    for (int j = 0; j < vectors[i].width; j++) {
                        result[i][j] = vectors[i].get(j);
                    }
                }
            } else {
                result = new double[vectors[0].height][vectors.length];
                for (int i = 0; i < vectors.length; i++) {
                    for (int j = 0; j < vectors[i].height; j++) {
                        result[j][i] = vectors[i].get(j);
                    }
                }
            }
            return Tensor.build(result);
        }
    }

    private double multiplyMatricesCell(Tensor first, Tensor second, int row, int col) {
        double cell = 0;
        for (int i = 0; i < second.height; i++) {
            cell += first.get(row, i) * second.get(i, col);
        }
        return cell;
    }
}
