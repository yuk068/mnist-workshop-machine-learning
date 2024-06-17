package mnist.training;

import mnist.utility.ActivationFunction;
import mnist.utility.MNISTReader;

import java.io.IOException;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The NeuralNetwork class represents a neural network composed of layers of neurons.
 * It includes methods for adding layers, training, testing, and making predictions.
 */
public class NeuralNetwork implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final List<NeuronLayer> layers;
    private final ActivationFunction outputActivation;

    /**
     * Constructs a NeuralNetwork with the specified output activation function.
     *
     * @param outputActivation the activation function to be used by the output layer
     */
    public NeuralNetwork(ActivationFunction outputActivation) {
        layers = new ArrayList<>();
        this.outputActivation = outputActivation;
    }

    /**
     * Adds a hidden layer to the neural network.
     *
     * @param numNeurons the number of neurons in the hidden layer
     * @param inputSize the number of inputs to each neuron in the hidden layer
     * @param activation the activation function to be used by the hidden layer
     */
    public void addHiddenLayer(int numNeurons, int inputSize, ActivationFunction activation) {
        layers.add(new NeuronLayer(numNeurons, inputSize, activation, false));
    }

    /**
     * Adds an output layer to the neural network.
     *
     * @param outputClassificationSize the number of neurons in the output layer
     * @param inputSize the number of inputs to each neuron in the output layer
     */
    public void addOutputLayer(int outputClassificationSize, int inputSize) {
        layers.add(new NeuronLayer(outputClassificationSize, inputSize, outputActivation, true));
    }

    /**
     * Performs a forward pass through the network up to the specified layer.
     *
     * @param inputs the input values to the network
     * @param layerIndex the index of the layer up to which the forward pass is performed
     * @return the output values from the specified layer
     */
    public double[] forwardPass(double[] inputs, int layerIndex) {
        double[] outputs = layers.get(0).forward(inputs);
        for (int i = 1; i < layerIndex; i++) {
            outputs = layers.get(i).forward(outputs);
        }
        return outputs;
    }

    /**
     * Performs a backward pass through the network to calculate the deltas (errors).
     *
     * @param target the target output values
     * @param predicted the predicted output values
     * @return the calculated deltas for each layer
     */
    public double[][] backwardPass(double[] target, double[] predicted) {
        int numLayers = layers.size();
        double[][] deltas = new double[numLayers][];
        double[] error = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            error[i] = predicted[i] - target[i];
        }

        deltas[numLayers - 1] = error;

        for (int l = numLayers - 2; l >= 0; l--) {
            NeuronLayer currentLayer = layers.get(l);
            NeuronLayer nextLayer = layers.get(l + 1);
            double[] currentDelta = new double[currentLayer.getNumNeurons()];
            for (int i = 0; i < currentLayer.getNumNeurons(); i++) {
                double deltaSum = 0.0;
                for (int j = 0; j < nextLayer.getNumNeurons(); j++) {
                    deltaSum += nextLayer.getWeights()[j][i] * deltas[l + 1][j];
                }
                currentDelta[i] = deltaSum * currentLayer.getActivation().
                        derivative(currentLayer.getOutputs()[i]);
            }
            deltas[l] = currentDelta;
        }
        return deltas;
    }

    /**
     * Calculates the gradients for all layers based on the inputs and deltas.
     *
     * @param inputs the input values to the network
     * @param deltas the calculated deltas for each layer
     * @return the calculated gradients for each layer
     */
    public double[][][] calculateGradients(double[] inputs, double[][] deltas) {
        int numLayers = layers.size();
        double[][][] gradients = new double[numLayers][][];

        double[][] activations = new double[numLayers + 1][];
        activations[0] = inputs;

        for (int i = 0; i < numLayers; i++) {
            activations[i + 1] = forwardPass(inputs, i);
        }

        for (int i = 0; i < numLayers; i++) {
            NeuronLayer currentLayer = layers.get(i);
            int numNeurons = currentLayer.getNumNeurons();
            int inputSize = currentLayer.getInputSize();

            gradients[i] = new double[numNeurons][inputSize + 1];

            for (int j = 0; j < numNeurons; j++) {
                for (int k = 0; k < inputSize; k++) {
                    gradients[i][j][k] = deltas[i][j] * activations[i][k];
                }
                gradients[i][j][inputSize] = deltas[i][j];
            }
        }

        return gradients;
    }

    /**
     * Updates the parameters of all layers based on the calculated gradients and learning rate.
     *
     * @param gradients the calculated gradients for each layer
     * @param learningRate the learning rate to be used for updating parameters
     */
    public void updateParams(double[][][] gradients, double learningRate) {
        int numLayers = layers.size();
        for (int l = 0; l < numLayers; l++) {
            NeuronLayer layer = layers.get(l);
            layer.updateParams(gradients[l], learningRate);
        }
    }

    /**
     * Performs backpropagation to update the network's parameters based on the inputs, target output, and learning rate.
     *
     * @param inputs the input values to the network
     * @param target the target output values
     * @param learningRate the learning rate to be used for updating parameters
     */
    public void backPropagation(double[] inputs, double[] target, double learningRate) {
        double[] predicted = forwardPass(inputs, layers.size());
        double[][] deltas = backwardPass(target, predicted);
        double[][][] gradients = calculateGradients(inputs, deltas);
        updateParams(gradients, learningRate);
    }

    /**
     * Calculates the cross-entropy loss between the output and target values.
     *
     * @param output the output values from the network
     * @param target the target output values
     * @return the calculated cross-entropy loss
     */
    public double calculateCrossEntropyLoss(double[] output, double[] target) {
        double loss = 0.0;

        for (int i = 0; i < 10; i++) {
            loss += target[i] * Math.log(output[i] + 1e-10);
        }
        return -loss;
    }

    /**
     * Trains the neural network using the MNIST dataset.
     *
     * @param labelFilePath the path to the label file
     * @param imageFilePath the path to the image file
     * @param learningRate the learning rate for training
     * @param epochs the number of epochs to train for
     * @param batchSize the batch size for training
     * @param flashRead flag indicating whether to use flash read for the dataset
     * @throws IOException if an I/O error occurs during reading the dataset
     */
    public void train(String labelFilePath, String imageFilePath, double learningRate, int epochs, int batchSize, boolean flashRead) throws IOException {
        int[] labels;
        int[][][] images;
        if (flashRead) {
            labels = MNISTReader.flashReadLabels(labelFilePath, batchSize);
            images = MNISTReader.flashReadImages(imageFilePath, batchSize);
        } else {
            labels = MNISTReader.readLabels(labelFilePath, batchSize);
            images = MNISTReader.readImages(imageFilePath, batchSize);
        }

        System.out.println("Training started...");
        System.out.println("Epochs: " + epochs);
        System.out.println("Learning rate: " + learningRate);
        System.out.println("Number of training examples: " + labels.length);

        Random rand = new Random();

        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffleDataset(labels, images, rand);

            double totalLoss = 0.0;

            for (int i = 0; i < labels.length; i++) {
                double[] input = preprocessImage(images[i]);
                double[] target = preprocessLabel(labels[i]);

                backPropagation(input, target, learningRate);

                double[] output = forwardPass(input, layers.size());
                double loss = calculateCrossEntropyLoss(output, target);
                totalLoss += loss;
            }

            System.out.println("Epoch " + (epoch + 1) + " Loss: " + (totalLoss / labels.length));
        }

        System.out.println("Training completed.");
    }

    /**
     * Shuffles the dataset to ensure random distribution of training examples.
     *
     * @param labels the array of labels
     * @param images the array of images
     * @param rand the random number generator used for shuffling
     */
    public void shuffleDataset(int[] labels, int[][][] images, Random rand) {
        for (int i = labels.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);

            int tempLabel = labels[i];
            labels[i] = labels[j];
            labels[j] = tempLabel;

            int[][] tempImage = images[i];
            images[i] = images[j];
            images[j] = tempImage;
        }
    }

    /**
     * Tests the neural network using the MNIST dataset.
     *
     * @param labelFilePath the path to the label file
     * @param imageFilePath the path to the image file
     * @param batchSize the batch size for testing
     * @param flashRead flag indicating whether to use flash read for the dataset
     * @throws IOException if an I/O error occurs during reading the dataset
     */
    public void test(String labelFilePath, String imageFilePath, int batchSize, boolean flashRead) throws IOException {
        int[] labels;
        int[][][] images;
        if (flashRead) {
            labels = MNISTReader.flashReadLabels(labelFilePath, batchSize);
            images = MNISTReader.flashReadImages(imageFilePath, batchSize);
        } else {
            labels = MNISTReader.readLabels(labelFilePath, batchSize);
            images = MNISTReader.readImages(imageFilePath, batchSize);
        }

        System.out.println("Testing started...");
        System.out.println("Number of test examples: " + labels.length);

        int correctPredictions = 0;

        for (int i = 0; i < labels.length; i++) {
            double[] input = preprocessImage(images[i]);
            int predictedLabel = predict(input);
            if (predictedLabel == labels[i]) {
                correctPredictions++;
            }
        }

        double accuracy = (double) correctPredictions / labels.length;
        System.out.println("Testing completed. Accuracy: " + (accuracy * 100) + "%");
    }

    /**
     * Predicts the label of the given inputs using the neural network.
     *
     * @param inputs the input values to the network
     * @return the predicted label
     */
    public int predict(double[] inputs) {
        double[] output = forwardPass(inputs, layers.size());
        int predictedLabel = 0;
        double maxOutput = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxOutput) {
                predictedLabel = i;
                maxOutput = output[i];
            }
        }

        return predictedLabel;
    }

    /**
     * Predicts the label and confidence of the given image.
     *
     * @param imagePath the path to the image file
     * @return an array containing the predicted label and confidence
     */
    public double[] predict(String imagePath) {
        int[][] image = MNISTReader.loadImage(imagePath);
        double[] inputs = preprocessImage(image);
        double[] output = forwardPass(inputs, layers.size());
        int predictedLabel = 0;
        double maxOutput = output[0];
        for (int i = 1; i < output.length; i++) {
            if (output[i] > maxOutput) {
                predictedLabel = i;
                maxOutput = output[i];
            }
        }

        return new double[]{predictedLabel, maxOutput};
    }

    /**
     * Preprocesses the image by normalizing the pixel values.
     *
     * @param image the image to be preprocessed
     * @return the normalized pixel values as a 1D array
     */
    public double[] preprocessImage(int[][] image) {
        double[] input = new double[image.length * image[0].length];
        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[i].length; j++) {
                input[i * image[i].length + j] = image[i][j] / 255.0;
            }
        }
        return input;
    }

    /**
     * Preprocesses the label by creating a one-hot encoded vector.
     *
     * @param label the label to be preprocessed
     * @return the one-hot encoded vector
     */
    public double[] preprocessLabel(int label) {
        double[] target = new double[10];
        target[label] = 1.0;
        return target;
    }

    /**
     * Returns a string representation of the neural network, including details of each layer.
     *
     * @return a string representation of the neural network
     */
    @Override
    public String toString() {
        StringBuilder network = new StringBuilder();
        int index = 0;
        for (NeuronLayer layer : layers) {
            network.append("Layer ").
                    append(index + 1).
                    append(": ").
                    append(layer).
                    append("\n");
            index++;
        }
        return network.toString();
    }

    /**
     * Prints the details of all neurons in the specified layer to the console.
     *
     * @param index the index of the layer
     */
    public void printNeuronsOfLayer(int index) {
        System.out.println("Layer " + (index + 1) + ": ");
        layers.get(index).printAllNeurons();
    }

    /**
     * Prints the details of all neurons in all layers to the console.
     */
    public void printAllNeurons() {
        layers.forEach(NeuronLayer::printAllNeurons);
    }

    /**
     * Returns the list of neuron layers in the neural network.
     *
     * @return the list of neuron layers
     */
    public List<NeuronLayer> getLayers() {
        return layers;
    }
    
}
