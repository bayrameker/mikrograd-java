package com.mikrograd;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    List<Layer> layers;

    public NeuralNetwork(int[] layerSizes) {
        layers = new ArrayList<>();
        for (int i = 0; i < layerSizes.length - 1; i++) {
            layers.add(new Layer(layerSizes[i + 1], layerSizes[i]));
        }
    }

    public List<Double> predict(List<Double> inputs) {
        List<Double> outputs = inputs;
        for (Layer layer : layers) {
            outputs = layer.forward(outputs);
        }
        return outputs;
    }

    public void train(List<List<Double>> trainingData, List<Double> labels, double learningRate, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainingData.size(); i++) {
                List<Double> inputs = trainingData.get(i);
                double label = labels.get(i);
                List<Double> outputs = predict(inputs);

                // Backpropagation
                List<Double> deltas = new ArrayList<>();
                for (int j = 0; j < outputs.size(); j++) {
                    double error = outputs.get(j) - label;
                    deltas.add(error * sigmoidDerivative(outputs.get(j)));
                }

                for (int j = layers.size() - 1; j >= 0; j--) {
                    Layer layer = layers.get(j);
                    List<Double> newDeltas = new ArrayList<>();
                    for (int k = 0; k < layer.neurons.size(); k++) {
                        Neuron neuron = layer.neurons.get(k);
                        for (int l = 0; l < neuron.weights.size(); l++) {
                            double input = 0.0;
                            if (j == 0) {
                                input = inputs.get(l);
                            } else {
                                input = layers.get(j - 1).neurons.get(l).output;
                            }
                            neuron.weights.set(l, neuron.weights.get(l) - learningRate * deltas.get(k) * input);
                        }
                        neuron.bias -= learningRate * deltas.get(k);

                        if (j > 0) {
                            for (int l = 0; l < neuron.weights.size(); l++) {
                                if (l < layers.get(j - 1).neurons.size()) {
                                    double newDelta = deltas.get(k) * neuron.weights.get(l) * sigmoidDerivative(layers.get(j - 1).neurons.get(l).output);
                                    if (newDeltas.size() <= l) {
                                        newDeltas.add(newDelta);
                                    } else {
                                        newDeltas.set(l, newDeltas.get(l) + newDelta);
                                    }
                                }
                            }
                        }
                    }
                    deltas = newDeltas;
                }
            }
        }
    }

    private double sigmoidDerivative(double z) {
        return z * (1 - z);
    }
}
