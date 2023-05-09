package ai.test.algebra;

import lombok.Getter;

@Getter
public class Shape {

    private final int x;

    private final int y;

    private final int z;

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

    @Override
    public int hashCode() {
        return 31 * x + 191 * y + 401 * z;
    }

    @Override
    public boolean equals(Object obj) {
        if (obj.getClass() != Shape.class) {
            return false;
        } else {
            Shape shapeObj = (Shape) obj;
            return this.x == shapeObj.x && this.y == shapeObj.y && this.z == shapeObj.z;
        }
    }
}
