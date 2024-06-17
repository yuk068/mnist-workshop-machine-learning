package mnist.training;

import mnist.utility.ActivationFunction;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Random;

/**
 * The Neuron class represents a single neuron in a neural network.
 * It includes the neuron's weights, bias, activation function, and methods
 * for performing forward propagation and parameter updates.
 */
public class Neuron implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private double bias;
    private final double[] weights;
    private final ActivationFunction activation;
    private double output;

    /**
     * Constructs a Neuron with the specified number of inputs and activation function.
     *
     * @param inputSize the number of inputs to the neuron
     * @param function the activation function to be used by the neuron
     */
    public Neuron(int inputSize, ActivationFunction function) {
        weights = new double[inputSize];
        activation = function;
        initializeParams();
    }

    /**
     * Initializes the weights and bias of the neuron with random values.
     * The weights are initialized using a Gaussian distribution scaled by the
     * inverse square root of the number of inputs.
     */
    public void initializeParams() {
        Random rand = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextGaussian() / Math.sqrt(weights.length);
        }
        bias = rand.nextGaussian() * 0.1;
    }

    /**
     * Performs forward propagation by calculating the weighted sum of inputs,
     * adding the bias, and applying the activation function.
     *
     * @param inputs the input values to the neuron
     * @return the output of the neuron after applying the activation function
     * @throws IllegalArgumentException if the number of inputs does not match the number of weights
     */
    public double forward(double[] inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException("Number of inputs must match number of weights.");
        }

        double z = bias;
        for (int i = 0; i < inputs.length; i++) {
            z += weights[i] * inputs[i];
        }
        output = activation.apply(z);
        return output;
    }

    /**
     * Returns the weights of the neuron.
     *
     * @return an array of weights
     */
    public double[] getWeights() {
        return weights;
    }

    /**
     * Returns the output of the neuron after the last forward propagation.
     *
     * @return the output value
     */
    public double getOutput() {
        return output;
    }

    /**
     * Returns the bias of the neuron.
     *
     * @return the bias value
     */
    public double getBias() {
        return bias;
    }

    /**
     * Updates the weights and bias of the neuron using the provided gradients
     * and learning rate. The last element of the gradients array is used to
     * update the bias.
     *
     * @param gradients the gradients for the weights and bias
     * @param learningRate the learning rate to be used for updating parameters
     * @throws IllegalArgumentException if the number of gradients does not match the number of weights + 1
     */
    public void updateParams(double[] gradients, double learningRate) {
        if (gradients.length - 1 != weights.length) {
            throw new IllegalArgumentException("Number of gradients must match number of weights.");
        }

        for (int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * gradients[i];
        }
        bias -= learningRate * gradients[weights.length];
    }

    /**
     * Returns a string representation of the neuron, including its weights and bias.
     *
     * @return a string representation of the neuron
     */
    @Override
    public String toString() {
        return "Neuron:" +
                "[weights=" + Arrays.toString(weights) +
                ", bias=" + bias +
                ']';
    }

}
