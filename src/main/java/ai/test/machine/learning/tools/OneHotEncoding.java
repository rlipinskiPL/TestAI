package ai.test.machine.learning.tools;

import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;

import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;

public class OneHotEncoding implements Tool {

    private final Map<Double, Integer> labels = new HashMap<>();

    @Override
    public void fit(Tensor X) {
        if (!X.isVector()) {
            throw new IllegalArgumentException("Only vectors are acceptable in OneHotEncoding");
        }

        Vector Xvector = (Vector) X;
        int size = Math.max(Xvector.height(), Xvector.width());

        for (int i = 0; i < size; i++) {
            Double value = Xvector.get(i);
            if (!labels.containsKey(value)) {
                labels.put(value, labels.size());
            }
        }
    }

    @Override
    public Tensor transform(Tensor X) {
        if (!X.isVector()) {
            throw new IllegalArgumentException("Only vectors are acceptable in OneHotEncoding");
        }

        Vector Xvector = (Vector) X;
        int size = Math.max(Xvector.height(), Xvector.width());
        Tensor result = Tensor.build(new double[size][labels.size()]);

        for (int i = 0; i < size; i++) {
            Integer label = labels.get(Xvector.get(i));
            if (label != null) {
                result.set(i, label, 1);
            } else {
                throw new NoSuchElementException("Label " + Xvector.get(i) + " was not present during fit method execution");
            }
        }

        return result;
    }
}
