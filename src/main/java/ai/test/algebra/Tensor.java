package ai.test.algebra;

import lombok.Getter;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Stream;

/**
 * This abstract class is a base class for all tensors classes.
 * It doesn't contain any objects for storing values, this responsibility
 * lie with child classes.
 * It provides static methods to create tensors in different ways and
 * some abstract methods to be implemented in subclasses.
 */
public abstract class Tensor implements Cloneable {

    protected int width;

    protected int height;

    @Getter
    protected Shape shape;

    protected Tensor(int height, int width) {
        if (height < 0 || width < 0) {
            throw new IllegalArgumentException("Dimensions of the tensor cannot be negative numbers");
        }
        this.width = width;
        this.height = height;
        this.shape = new Shape(height, width);
    }

    /**
     * This method performs transpose operation.
     *
     * @return transposed tensor
     */
    public abstract Tensor transpose();

    /**
     * This method performs multiplication by scalar operation.
     *
     * @param value scalar by which tensor is to be multiplied
     * @return multiplied tensor
     */
    public abstract Tensor multiply(double value);

    /**
     * This method add scalar to all tensor elements.
     *
     * @param value scalar to be added to tensor
     * @return tensor with elements enlarged by scalar
     */
    public abstract Tensor addition(double value);

    /**
     * This method raise all elements of tensor to a given power
     *
     * @param value exponent
     * @return tensor with elements raised to a given power
     */
    public abstract Tensor power(double value);

    /**
     * This method returns specific element of tensor.
     *
     * @param i row
     * @param j column
     * @return value of element at [i,j] position
     */
    public abstract double get(int i, int j);

    /**
     * This method set value of a specific element in tensor.
     *
     * @param i     row
     * @param j     column
     * @param value new value of an element
     */
    public abstract void set(int i, int j, double value);

    /**
     * If tensor is 1x1 size this method return this only value.
     *
     * @return The only value in tensor
     * @throws IllegalStateException when tensor has more than one element
     */
    public abstract double getAsScalar() throws IllegalStateException;

    /**
     * This method diminish size of tensor. Result of this method depends on implementation in subclasses.
     *
     * @param start position of first element that is included in resulting tensor
     * @param end   position of last element that is included in resulting tensor
     * @return diminished tensor
     */
    public abstract Tensor cut(int start, int end);

    /**
     * This method prints tensor to standard output.
     */
    public abstract void print();

    /**
     * This method returns stream over elements in tensor.
     *
     * @return stream over elements in tensor
     */
    public abstract Stream<?> stream();

    @Override
    public abstract java.lang.Object clone();

    /**
     * This method performs matrix multiplication AxB. Where A is this object and B is parameter.
     *
     * @param tensor second factor of multiplication
     * @return resulting tensor
     * @throws ArithmeticException when tensors sizes do not allow multiplication
     */
    public Tensor matmul(Tensor tensor) {
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

    /**
     * This method performs elementwise multiplication this object with parameter.
     *
     * @param tensor factor of multiplication
     * @return resulting tensor
     * @throws ArithmeticException when shape of two factors are not the same
     */
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

    /**
     * This method performs addition this tensor with parameter.
     *
     * @param tensor addend
     * @return resulting tensor
     * @throws ArithmeticException when shape of two factors are not the same
     */
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

    /**
     * This method performs substraction operation, where this object is minuend and parameter is subtrahend.
     *
     * @param tensor subtrahend
     * @return resulting tensor
     * @throws ArithmeticException when shape of two factors are not the same
     */
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

    /**
     * @return if this object is vector
     */
    public boolean isVector() {
        return width == 1 || height == 1;
    }

    /**
     * @return height of tensor
     */
    public int height() {
        return height;
    }

    /**
     * @return width of tensor
     */
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

    /**
     * As elements of a tensor are stored as a double sometimes two theoretically equal values
     * are not equal in terms of equal() methods due to finite precision. So this method do the same
     * thing as equal() but allow some error to avoid above situation.
     *
     * @param obj       object to check if equals to this object
     * @param precision error that is allowed, for example when error is 0.2 this values: 2.1 and 2.2 are equal
     * @return if this parameter is equal to this object
     */
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

    /**
     * Build tensor from nested lists of doubles
     *
     * @param values nested list of doubles
     * @return built tensor
     * @throws IllegalArgumentException when main list is empty
     */
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

    /**
     * Build tensor from two-dimensional array of doubles
     *
     * @param values two-dimensional array of doubles
     * @return built tensor
     * @throws IllegalArgumentException when main array is empty
     */
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

    /**
     * Build tensor from array of doubles
     *
     * @param values     array of doubles
     * @param horizontal if true built vector has shape 1xn, else nx1
     * @return built tensor
     * @throws IllegalArgumentException when array is empty
     */
    public static Tensor build(double[] values, boolean horizontal) {
        if (values.length == 0) {
            throw new IllegalArgumentException("Empty array is not permitted here");
        }
        return new Vector(values, horizontal);
    }

    /**
     * Build tensor from list of doubles
     *
     * @param values     list of doubles
     * @param horizontal if true built vector has shape 1xn, else nx1
     * @return built tensor
     * @throws IllegalArgumentException when list is empty
     */
    public static Tensor build(List<Double> values, boolean horizontal) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("Empty list is not permitted here");
        }
        return new Vector(values, horizontal);
    }

    /**
     * Build tensor from doubles given as args
     *
     * @param values     args of doubles
     * @param horizontal if true built vector has shape 1xn, else nx1
     * @return built tensor
     */
    public static Tensor build(boolean horizontal, double... values) {
        return Tensor.build(values, horizontal);
    }

    /**
     * This method build unitary matrix of given size
     *
     * @param dimension size of matrix
     * @return unitary matrix
     */
    public static Tensor eye(int dimension) {
        double[][] result = new double[dimension][dimension];
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                result[i][j] = i == j ? 1 : 0;
            }
        }
        return Tensor.build(result);
    }

    /**
     * This method built tensor from csv file.
     * File must not contain any headers or indexes, just plain data for tensor.
     *
     * @param filePath path to the csv file with data
     * @return built tensor
     */
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

    /**
     * This method build matrix from list of vectors.
     *
     * @param vectors
     * @return built matrix
     * @throws IllegalArgumentException when list is empty or vectors don't have the same size
     */
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
