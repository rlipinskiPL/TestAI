package org.example.algebra;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class Vector extends Tensor {
    private double[] data;
    private boolean isHorizontal;

    public Vector(List<Double> values, boolean horizontal){
        super(horizontal ? 1 : values.size(), horizontal ? values.size() : 1);
        this.isHorizontal=horizontal;
        data = new double[values.size()];
        for(int i=0;i< values.size();i++){
            data[i] = values.get(i);
        }
    }

    public Vector(double[] values, boolean horizontal){
        super(horizontal ? 1 : values.length, horizontal ? values.length : 1);
        this.isHorizontal=horizontal;
        data=values.clone();
    }

    @Override
    public double get(int i, int j){
        if((isHorizontal && i>0) || (!isHorizontal && j>0)){
            throw new ArrayIndexOutOfBoundsException();
        }

        if(isHorizontal){
            return data[j];
        }else{
            return data[i];
        }
    }

    @Override
    public void set(int i, int j, double value){
        if((isHorizontal && i>0) || (!isHorizontal && j>0)){
            throw new ArrayIndexOutOfBoundsException();
        }

        if(isHorizontal){
            data[j] = value;
        }else{
            data[i] = value;
        }
    }

    @Override
    public double getAsScalar(){
        if(data.length == 1){
            return data[0];
        }else{
            throw new IllegalStateException("Cannot parse to scalar because vector size is not 1");
        }
    }

    public double get(int i){
        return data[i];
    }

    public void set(int i, double value){
        data[i] = value;
    }

    @Override
    public Tensor transpose() {
        return new Vector(data.clone(),!isHorizontal);
    }

    @Override
    public Tensor multiply(double value) {
        return Tensor.build(Arrays.stream(data).map(o->value*o).toArray(),isHorizontal);
    }

    @Override
    public Tensor addition(double value) {
        return Tensor.build(Arrays.stream(data).map(o->value+o).toArray(),isHorizontal);
    }

    @Override
    public Tensor power(double value) {
        return Tensor.build(Arrays.stream(data).map(o-> java.lang.Math.pow(o,value)).toArray(),isHorizontal);
    }

    @Override
    public Tensor cut(int start, int end){
        if (start > end) {
            throw new IllegalArgumentException("Start must be smaller value than end");
        }else if (start >= data.length) {
            throw new IllegalArgumentException("Start must be smaller than height of matrix");
        }

        int size = java.lang.Math.max(width,height);

        double[] result = end < size ? new double[end - start + 1] : new double[size - start + 1];
        for (int i = start, k = 0; i <= end; i++, k++) {
            if(i == data.length)
                break;

            result[k] = data[i];
        }
        return Tensor.build(result, isHorizontal);
    }

    @Override
    public void print(){
        if(isHorizontal){
            StringBuilder builder = new StringBuilder("[");
            Arrays.stream(data).forEachOrdered(o->builder.append(o+"|"));
            builder.deleteCharAt(builder.length()-1);
            builder.append("]");
            System.out.println(builder);
        }else{
            Arrays.stream(data).forEachOrdered(o-> System.out.println("["+o+"]"));
        }
    }

    @Override
    public java.lang.Object clone(){
        return Tensor.build(data.clone(),isHorizontal);
    }

    @Override
    public Stream<Double> stream(){
        return Arrays.stream(data).boxed();
    }

    public boolean isHorizontal(){
        return isHorizontal;
    }
}
