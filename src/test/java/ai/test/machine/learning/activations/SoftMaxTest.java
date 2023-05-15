package ai.test.machine.learning.activations;

import ai.test.algebra.Tensor;
import ai.test.algebra.Vector;
import org.junit.Test;
import org.junit.Assert;

import java.util.List;

public class SoftMaxTest {
    SoftMax softmax = new SoftMax();

    @Test
    public void testCallWithVector_NormalValues1(){
        Tensor input = Tensor.build(List.of(1.,2.,3.),true);

        Tensor output = softmax.call(input);

        Tensor result = Tensor.build(List.of(0.09003057, 0.24472847, 0.66524096),true);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testCallWithVector_NormalValues2(){
        Tensor input = Tensor.build(List.of(.5,.5,.5),true);

        Tensor output = softmax.call(input);

        Tensor result = Tensor.build(List.of(0.33333333, 0.33333333, 0.33333333),true);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testCallWithVector_PositiveAndNegativeValues(){
        Tensor input = Tensor.build(List.of(-1., 0., 1.),true);

        Tensor output = softmax.call(input);

        Tensor result = Tensor.build(List.of(0.09003057, 0.24472847, 0.66524095),true);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testCallWithVector_BigNegativeValues(){
        Tensor input = Tensor.build(List.of(-100., -200., -300.),true);

        Tensor output = softmax.call(input);

        Tensor result = Tensor.build(List.of(1.,0.,0.),true);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testCallWithVector_BigValues1(){ //checking numerical stability
        Tensor input = Tensor.build(List.of(100., 200., 300.),true);

        Tensor output = softmax.call(input);

        Tensor result = Tensor.build(List.of(0., 0., 1.),true);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testCallWithVector_BigValues2(){ //checking numerical stability
        Tensor input = Tensor.build(List.of(10000., 20000., 30000.),true);

        Tensor output = softmax.call(input);

        Tensor result = Tensor.build(List.of(0., 0., 1.),true);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testCallWithVector_BigValues3(){ //checking numerical stability
        Tensor input = Tensor.build(List.of(100000., 200000., 300000.),true);

        Tensor output = softmax.call(input);

        Tensor result = Tensor.build(List.of(0., 0., 1.),true);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testCallWithMatrix(){
        Tensor input = Tensor.build(List.of(
                List.of(1.,2.,3.),
                List.of(.5,.5,.5),
                List.of(-1., 0., 1.),
                List.of(-100., -200., -300.),
                List.of(100., 200., 300.),
                List.of(10000., 20000., 30000.),
                List.of(100000., 200000., 300000.)
        ));

        Tensor output = softmax.call(input);

        Tensor result = Tensor.build(List.of(
                List.of(0.09003057, 0.24472847, 0.66524096),
                List.of(0.33333333, 0.33333333, 0.33333333),
                List.of(0.09003057, 0.24472847, 0.66524095),
                List.of(1.,0.,0.),
                List.of(0., 0., 1.),
                List.of(0., 0., 1.),
                List.of(0., 0., 1.)
        ));

        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testDerivativeWithVector_NormalValues1(){
        Tensor input = Tensor.build(List.of(1.,2.,3.),true);

        Tensor output = softmax.derivative(input);

        double[][] resultA = {
                {0.08192507, -0.02203304, -0.05989202},
                {-0.02203304,  0.18483645, -0.1628034},
                {-0.05989202, -0.1628034,   0.22269543}
        };
        Tensor result = Tensor.build(resultA);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testDerivativeWithVector_NormalValues2(){
        Tensor input = Tensor.build(List.of(.5,.5,.5),true);

        Tensor output = softmax.derivative(input);

        double[][] resultA = {
                { 0.22222222, -0.11111111, -0.11111111},
                { -0.11111111,  0.22222222, -0.11111111},
                {-0.11111111, -0.11111111,  0.22222222}
        };
        Tensor result = Tensor.build(resultA);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testDerivativeWithVector_PositiveAndNegativeValues(){
        Tensor input = Tensor.build(List.of(-1., 0., 1.),true);

        Tensor output = softmax.derivative(input);

        double[][] resultA = {
                { 0.08192507, -0.02203304, -0.05989202},
                {-0.02203304,  0.18483645, -0.1628034 },
                {-0.05989202, -0.1628034,   0.22269543}
        };
        Tensor result = Tensor.build(resultA);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testDerivativeWithVector_BigNegativeValues(){
        Tensor input = Tensor.build(List.of(-100., -200., -300.),true);

        Tensor output = softmax.derivative(input);

        double[][] resultA = {
                { 0.00000000e+000, -3.72007598e-044, -1.38389653e-087},
                {-3.72007598e-044,  3.72007598e-044, -5.14820022e-131},
                {-1.38389653e-087, -5.14820022e-131,  1.38389653e-087}
        };
        Tensor result = Tensor.build(resultA);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testDerivativeWithVector_BigValues1(){ //checking numerical stability
        Tensor input = Tensor.build(List.of(100., 200., 300.),true);

        Tensor output = softmax.derivative(input);

        double[][] resultA = {
                { 1.38389653e-087, -5.14820022e-131, -1.38389653e-087},
                {-5.14820022e-131,  3.72007598e-044, -3.72007598e-044},
                {-1.38389653e-087, -3.72007598e-044,  0.00000000e+000}
        };
        Tensor result = Tensor.build(resultA);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testDerivativeWithVector_BigValues2(){ //checking numerical stability
        Tensor input = Tensor.build(List.of(10000., 20000., 30000.),true);

        Tensor output = softmax.derivative(input);

        double[][] resultA = {
                { 0, 0, 0},
                {0,  0, 0},
                {0, 0,  0}
        };
        Tensor result = Tensor.build(resultA);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }

    @Test
    public void testDerivativeWithMatrix(){
        Tensor input = Tensor.build(List.of(
                List.of(1.,2.,3.),
                List.of(.5,.5,.5),
                List.of(-1., 0., 1.),
                List.of(-100., -200., -300.),
                List.of(100., 200., 300.),
                List.of(10000., 20000., 30000.)
        ));

        Tensor output = softmax.derivative(input);

        double[][] resultA = {
                {0.08192507, -0.02203304, -0.05989202},
                {-0.02203304,  0.18483645, -0.1628034},
                {-0.05989202, -0.1628034,   0.22269543},
                { 0.22222222, -0.11111111, -0.11111111},
                { -0.11111111,  0.22222222, -0.11111111},
                {-0.11111111, -0.11111111,  0.22222222},
                { 0.08192507, -0.02203304, -0.05989202},
                {-0.02203304,  0.18483645, -0.1628034 },
                {-0.05989202, -0.1628034,   0.22269543},
                { 0.00000000e+000, -3.72007598e-044, -1.38389653e-087},
                {-3.72007598e-044,  3.72007598e-044, -5.14820022e-131},
                {-1.38389653e-087, -5.14820022e-131,  1.38389653e-087},
                { 1.38389653e-087, -5.14820022e-131, -1.38389653e-087},
                {-5.14820022e-131,  3.72007598e-044, -3.72007598e-044},
                {-1.38389653e-087, -3.72007598e-044,  0.00000000e+000},
                { 0, 0, 0},
                {0,  0, 0},
                {0, 0,  0}
        };
        Tensor result = Tensor.build(resultA);
        Assert.assertTrue(output.equalsWithPrecision(result,0.0000001));
    }
}
