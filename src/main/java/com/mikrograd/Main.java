package com.mikrograd;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        int[] layerSizes = {2, 2, 1}; // 2 inputs, 1 hidden layer with 2 neurons, 1 output
        NeuralNetwork nn = new NeuralNetwork(layerSizes);

        // Training data (XOR problem)
        List<List<Double>> trainingData = Arrays.asList(
                Arrays.asList(0.0, 0.0),
                Arrays.asList(0.0, 1.0),
                Arrays.asList(1.0, 0.0),
                Arrays.asList(1.0, 1.0)
        );
        List<Double> labels = Arrays.asList(0.0, 1.0, 1.0, 0.0);

        nn.train(trainingData, labels, 0.1, 100000); // Increase epochs to ensure learning

        // Predictions
        for (List<Double> data : trainingData) {
            double output = nn.predict(data).get(0);
            System.out.println("Input: " + data + " -> Output: " + output);
        }
    }
}
