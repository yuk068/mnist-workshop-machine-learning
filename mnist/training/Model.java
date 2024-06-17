package mnist.training;

import java.io.*;
import java.util.List;

/**
 * The Model class represents a serializable neural network model for training, testing,
 * predicting, saving, and loading MNIST data.
 */
public class Model implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private final NeuralNetwork neuralNetwork;
    private final String modelName;

    /**
     * Constructs a Model with the specified name and neural network.
     *
     * @param name the name of the model
     * @param neuralNetwork the neural network associated with the model
     */
    public Model(String name, NeuralNetwork neuralNetwork) {
        modelName = name;
        this.neuralNetwork = neuralNetwork;
        validateNetwork();
    }

    /**
     * Trains the model using the specified parameters.
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
        System.out.println("Training operation for:");
        System.out.println(this);
        System.out.println("This might take awhile...\n");
        neuralNetwork.train(labelFilePath, imageFilePath, learningRate, epochs, batchSize, flashRead);
    }

    /**
     * Tests the model using the specified parameters.
     *
     * @param labelFilePath the path to the label file
     * @param imageFilePath the path to the image file
     * @param batchSize the batch size for testing
     * @param flashRead flag indicating whether to use flash read for the dataset
     * @throws IOException if an I/O error occurs during reading the dataset
     */
    public void test(String labelFilePath, String imageFilePath, int batchSize, boolean flashRead) throws IOException {
        System.out.println("Testing operation for:");
        System.out.println(this);
        System.out.println("This might take awhile...\n");
        neuralNetwork.test(labelFilePath, imageFilePath, batchSize, flashRead);
    }

    /**
     * Predicts the label for the given image path.
     *
     * @param imagePath the path to the image file
     * @return the predicted label
     */
    public int predict(String imagePath) {
        double[] result = neuralNetwork.predict(imagePath);

        String prediction = String.format(" Prediction: %d,", (int) result[0]);
        String confidence = String.format(" Confidence:%7.2f%%", result[1] * 100);

        System.out.print(prediction + confidence);

        return (int) result[0];
    }


    /**
     * Saves the model to a file.
     *
     * @throws IOException if an I/O error occurs during saving the model
     */
    public void save() throws IOException {
        String filePath = "src/mnist/model/";
        try (ObjectOutputStream outputStream = new ObjectOutputStream(new FileOutputStream(filePath + modelName + ".ser"))) {
            System.out.println("Saving model: " + modelName + "\n");
            outputStream.writeObject(this);
        }
    }

    /**
     * Loads a model from a file.
     *
     * @param modelName the name of the model to be loaded
     * @return the loaded Model object
     * @throws IOException if an I/O error occurs during loading the model
     * @throws ClassNotFoundException if the class of a serialized object cannot be found
     */
    public static Model load(String modelName) throws IOException, ClassNotFoundException {
        String filePath = "src/mnist/model/";
        try (ObjectInputStream inputStream = new ObjectInputStream(new FileInputStream(filePath + modelName + ".ser"))) {
            return (Model) inputStream.readObject();
        }
    }

    /**
     * Validates the neural network to ensure it meets the required configuration.
     *
     * @throws IllegalStateException if the network is not configured correctly
     */
    private void validateNetwork() {
        List<NeuronLayer> layers = neuralNetwork.getLayers();

        boolean hasIllegalNetwork = layers.size() < 2 ||
                !layers.get(layers.size() - 1).isLastLayer() ||
                layers.stream().filter(NeuronLayer::isLastLayer).count() != 1;

        if (hasIllegalNetwork) {
            throw new IllegalStateException("""
            Model's network is not configured correctly, please check again.
            Notable precautions:
            - At least 1 output layer, which must be the last layer.
            - At least 1 hidden layer.
            - Network must have at least 2 layers.
            """);
        }

        if (layers.get(0).getInputSize() != 784) {
            throw new IllegalStateException("Input layer must have 784 neurons corresponding to MNIST image size.");
        }

        for (int i = 1; i < layers.size(); i++) {
            NeuronLayer currentLayer = layers.get(i);
            NeuronLayer previousLayer = layers.get(i - 1);

            if (currentLayer.getInputSize() != previousLayer.getNumNeurons()) {
                throw new IllegalStateException("Layer " + i + " does not match previous layer in size.");
            }
        }

        NeuronLayer outputLayer = layers.get(layers.size() - 1);
        if (outputLayer.getNumNeurons() != 10) {
            throw new IllegalStateException("Output layer must have 10 neurons for classification.");
        }
    }

    /**
     * Returns a string representation of the network parameters including the number of weights and biases.
     *
     * @return a string representation of the network parameters
     */
    private String toStringNetworkParams() {
        List<NeuronLayer> layers = neuralNetwork.getLayers();
        int weights = 0;
        int biases = layers.get(0).getNumNeurons();
        for (int i = 1; i < layers.size(); i++) {
            weights += layers.get(i - 1).getInputSize() * layers.get(i).getInputSize();
            biases += layers.get(i).getNumNeurons();
        }
        weights += layers.get(layers.size() - 1).getInputSize() * layers.get(layers.size() - 1).getNumNeurons();

        return "{weights=" + weights + ", biases=" + biases + "}";
    }

    /**
     * Returns a string representation of the model including its name and network parameters.
     *
     * @return a string representation of the model
     */
    @Override
    public String toString() {
        return "MNIST Model \"" + modelName + "\": " +
                toStringNetworkParams() +
                "\n" + neuralNetwork;
    }

}
