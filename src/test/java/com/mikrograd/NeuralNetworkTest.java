package com.mikrograd;

import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import static org.junit.Assert.assertEquals;

public class NeuralNetworkTest {
    @Test
    public void testXorProblem() {
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
        for (int i = 0; i < trainingData.size(); i++) {
            List<Double> data = trainingData.get(i);
            double output = nn.predict(data).get(0);
            double expected = labels.get(i);
            System.out.println("Input: " + data + " -> Predicted: " + output + ", Expected: " + expected);
            assertEquals(expected, Math.round(output), 0.01); // Add a delta for floating point comparison
        }
    }
}
