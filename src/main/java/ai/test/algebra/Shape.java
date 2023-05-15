package ai.test.algebra;

import lombok.EqualsAndHashCode;
import lombok.Getter;

/**
 * This class represents shape of some object, so it brings information about dimensions and size in each dimension
 */
@Getter
@EqualsAndHashCode
public class Shape {

    /**
     * @return size of first dimension
     */
    private final int x;

    /**
     * @return size of second dimension
     */
    private final int y;

    /**
     * @return size of third dimension
     */
    private final int z;

    /**
     * @return number of dimension
     */
    private final int dimensions;

    public Shape(int x) {
        this.x = x;
        this.y = 0;
        this.z = 0;
        this.dimensions = 1;
    }

    public Shape(int x, int y) {
        this.x = x;
        this.y = y;
        this.z = 0;
        this.dimensions = 2;
    }

    public Shape(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.dimensions = 3;
    }
}
