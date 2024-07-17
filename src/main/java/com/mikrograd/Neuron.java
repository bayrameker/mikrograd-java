package com.mikrograd;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Neuron {
    List<Double> weights;
    double bias;
    double output;

    public Neuron(int numInputs) {
        Random rand = new Random();
        weights = new ArrayList<>();
        for (int i = 0; i < numInputs; i++) {
            weights.add(rand.nextGaussian() * 0.01); // Small random weights
        }
        bias = rand.nextGaussian() * 0.01; // Small random bias
    }

    public double activate(List<Double> inputs) {
        double z = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            z += inputs.get(i) * weights.get(i);
        }
        z += bias;
        output = sigmoid(z);
        return output;
    }

    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    public double sigmoidDerivative(double z) {
        return z * (1 - z);
    }
}
