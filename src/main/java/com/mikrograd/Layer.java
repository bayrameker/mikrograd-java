package com.mikrograd;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    List<Neuron> neurons;

    public Layer(int numNeurons, int numInputsPerNeuron) {
        neurons = new ArrayList<>();
        for (int i = 0; i < numNeurons; i++) {
            neurons.add(new Neuron(numInputsPerNeuron));
        }
    }

    public List<Double> forward(List<Double> inputs) {
        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : neurons) {
            outputs.add(neuron.activate(inputs));
        }
        return outputs;
    }
}
