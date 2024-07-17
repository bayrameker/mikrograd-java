

# Mikrograd Java Neural Network

This project implements a simple feedforward neural network in Java capable of learning the XOR function. The project consists of three main classes: `NeuralNetwork`, `Layer`, and `Neuron`, and includes a unit test to verify the network's ability to learn the XOR function.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Code Explanation](#code-explanation)
4. [How to Run](#how-to-run)
5. [Expected Output](#expected-output)
6. [Testing](#testing)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

The XOR (exclusive OR) function is a fundamental problem in machine learning and neural networks. It is often used to test the capabilities of a neural network to learn non-linear patterns. This project demonstrates a simple feedforward neural network trained using backpropagation to learn the XOR function.

## Project Structure

```
mikrograd-java/
├── src/
│   ├── main/
│   │   └── java/
│   │       └── com/
│   │           └── mikrograd/
│   │               ├── Layer.java
│   │               ├── Main.java
│   │               ├── NeuralNetwork.java
│   │               └── Neuron.java
│   └── test/
│       └── java/
│           └── com/
│               └── mikrograd/
│                   └── NeuralNetworkTest.java
├── pom.xml
└── README.md
```

## Code Explanation

### NeuralNetwork.java

This class represents the neural network and contains the main logic for forward propagation, prediction, and training using backpropagation.

```java
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
```

### Layer.java

This class represents a layer in the neural network and contains a list of neurons. It handles the forward propagation through the layer.

```java
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
```

### Neuron.java

This class represents a neuron in the neural network and contains the weights, bias, and output of the neuron. It handles the activation function and the computation of the neuron's output.

```java
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
```

### NeuralNetworkTest.java

This class contains the unit tests for the neural network, specifically testing the network's ability to learn the XOR function.

```java
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
```

### Main.java

This class is the entry point for running the neural network training and prediction from the command line.

```java
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
        List<Double> labels = Arrays.asList(0.0, 1

.0, 1.0, 0.0);

        nn.train(trainingData, labels, 0.1, 100000); // Increase epochs to ensure learning

        // Predictions
        for (List<Double> data : trainingData) {
            double output = nn.predict(data).get(0);
            System.out.println("Input: " + data + " -> Output: " + output);
        }
    }
}
```

## How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/bayrameker/mikrograd-java.git
   cd mikrograd-java
   ```

2. **Build the project:**

   ```bash
   mvn clean install
   ```

3. **Run the Main class:**

   ```bash
   mvn exec:java -Dexec.mainClass="com.mikrograd.Main"
   ```

4. **Run the tests:**

   ```bash
   mvn test
   ```

## Expected Output

When you run the tests, you should see output similar to the following:

```text
[INFO]  T E S T S
[INFO] -------------------------------------------------------
[INFO] Running com.mikrograd.NeuralNetworkTest
Input: [0.0, 0.0] -> Predicted: 0.013017411209498465, Expected: 0.0
Input: [0.0, 1.0] -> Predicted: 0.9851240479529629, Expected: 1.0
Input: [1.0, 0.0] -> Predicted: 0.9876308798642104, Expected: 1.0
Input: [1.0, 1.0] -> Predicted: 0.01166108596979877, Expected: 0.0
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 0.225 s -- in com.mikrograd.NeuralNetworkTest
[INFO] 
[INFO] Results:
[INFO] 
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0
[INFO] 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD SUCCESS
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  2.102 s
[INFO] Finished at: 2024-07-17T13:34:22+03:00
[INFO] ------------------------------------------------------------------------

Process finished with exit code 0
```

## Testing

This project includes unit tests to verify the functionality of the neural network. The `NeuralNetworkTest` class tests the network's ability to learn the XOR function. To run the tests, use the following command:

```bash
mvn test
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
