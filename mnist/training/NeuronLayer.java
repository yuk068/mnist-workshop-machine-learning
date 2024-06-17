package mnist.training;

import mnist.utility.ActivationFunction;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/**
 * The NeuronLayer class represents a layer of neurons in a neural network.
 * It includes the neurons, activation function, and methods for forward propagation
 * and parameter updates.
 */
public class NeuronLayer implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final Neuron[] neurons;
    private final ActivationFunction activation;
    private final boolean lastLayer;
    private final int inputSize;

    /**
     * Constructs a NeuronLayer with the specified number of neurons, input size, activation function,
     * and a flag indicating if it is the last layer.
     *
     * @param numNeurons the number of neurons in the layer
     * @param inputSize the number of inputs to each neuron
     * @param activation the activation function to be used by the neurons
     * @param lastLayer flag indicating if this is the last layer
     */
    public NeuronLayer(int numNeurons, int inputSize, ActivationFunction activation, boolean lastLayer) {
        this.neurons = new Neuron[numNeurons];
        this.activation = activation;
        this.lastLayer = lastLayer;
        this.inputSize = inputSize;
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(inputSize, activation);
        }
    }

    /**
     * Performs forward propagation for the entire layer.
     * If it is the last layer, it applies the activation function to the weighted sums
     * to produce the output probabilities.
     *
     * @param inputs the input values to the layer
     * @return the output values from the layer
     * @throws IllegalArgumentException if the number of inputs does not match the number of weights in the neurons
     */
    public double[] forward(double[] inputs) {
        double[] outputs = new double[neurons.length];

        if (lastLayer) {
            double[] yHat = new double[neurons.length];
            for (int i = 0; i < neurons.length; i++) {
                yHat[i] = neurons[i].getBias();
                for (int j = 0; j < inputs.length; j++) {
                    yHat[i] += neurons[i].getWeights()[j] * inputs[j];
                }
            }
            double[] probabilities = activation.apply(yHat);
            System.arraycopy(probabilities, 0, outputs, 0, outputs.length);
        } else {
            for (int i = 0; i < neurons.length; i++) {
                outputs[i] = neurons[i].forward(inputs);
            }
        }

        return outputs;
    }

    /**
     * Updates the weights and biases of all neurons in the layer using the provided gradients
     * and learning rate.
     *
     * @param gradients the gradients for the weights and biases of each neuron
     * @param learningRate the learning rate to be used for updating parameters
     * @throws IllegalArgumentException if the number of gradient arrays does not match the number of neurons
     */
    public void updateParams(double[][] gradients, double learningRate) {
        if (gradients.length != neurons.length) {
            throw new IllegalArgumentException("Number of gradient arrays must match number of neurons.");
        }
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].updateParams(gradients[i], learningRate);
        }
    }

    /**
     * Returns the weights of all neurons in the layer.
     *
     * @return a 2D array of weights, where each sub-array represents the weights of a neuron
     */
    public double[][] getWeights() {
        double[][] weights = new double[neurons.length][];
        for (int i = 0; i < neurons.length; i++) {
            weights[i] = neurons[i].getWeights();
        }
        return weights;
    }

    /**
     * Returns the number of neurons in the layer.
     *
     * @return the number of neurons
     */
    public int getNumNeurons() {
        return neurons.length;
    }

    /**
     * Returns the activation function used by the neurons in the layer.
     *
     * @return the activation function
     */
    public ActivationFunction getActivation() {
        return activation;
    }

    /**
     * Returns the number of inputs to each neuron in the layer.
     *
     * @return the number of inputs
     */
    public int getInputSize() {
        return inputSize;
    }

    /**
     * Returns whether this layer is the last layer in the network.
     *
     * @return true if this is the last layer, false otherwise
     */
    public boolean isLastLayer() {
        return lastLayer;
    }

    /**
     * Returns the outputs of all neurons in the layer after the last forward propagation.
     *
     * @return an array of output values
     */
    public double[] getOutputs() {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].getOutput();
        }
        return outputs;
    }

    /**
     * Returns a string representation of the NeuronLayer, including input size, number of neurons, and activation function.
     *
     * @return a string representation of the NeuronLayer
     */
    @Override
    public String toString() {
        return "{inputs=" + inputSize +
                ", neurons=" + getNumNeurons() +
                ", activation=" + activation +
                '}';
    }

    /**
     * Prints the details of all neurons in the layer to the console.
     */
    public void printAllNeurons() {
        for (Neuron neuron : neurons) {
            System.out.println(neuron);
        }
    }

}
