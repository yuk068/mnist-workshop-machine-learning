package mnist.training;

import mnist.utility.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MNISTWorkshop {

    private static final String TRAIN_LABEL_FILE = MNISTReader.TRAIN_LABEL_FILE;
    private static final String TRAIN_IMAGE_FILE = MNISTReader.TRAIN_IMAGE_FILE;
    private static final String TEST_LABEL_FILE = MNISTReader.TEST_LABEL_FILE;
    private static final String TEST_IMAGE_FILE = MNISTReader.TEST_IMAGE_FILE;

    private static final int TRAINING_BATCH_SIZE = 60000;
    private static final int TESTING_BATCH_SIZE = 10000;
    private static final double LEARNING_RATE = 0.001;
    private static final int EPOCHS = 5;

    final static boolean FLASH = false; // To read directly from disk or read cached data

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        /*
         * Pre-trained models:
         * Primus : relatively small model with Sigmoid(x)
         * PReLUs : same as Primus but uses ReLU(x)
         * Strata : larger model with 2 hidden layers, uses Sigmoid(x)
         * StraRe : same as Strata but uses ReLU(x)
         * Nexus  : 3 hidden layers, uses ReLU(x)
         * Prtanh : similar to Primus but uses Tanh(x)
         */

        Model primus = Model.load("Primus");
        Model prelus = Model.load("PReLUs");
        Model strata = Model.load("Strata");
        Model strare = Model.load("StraRe");
        Model nexus = Model.load("Nexus");
        Model prtanh = Model.load("Prtanh");
        Model teeny = Model.load("Teeny");

        List<Model> models = new ArrayList<>();
        models.add(primus);
        models.add(prelus);
        models.add(strata);
        models.add(strare);
        models.add(nexus);
        models.add(prtanh);
        models.add(teeny);

        testModelsFromPng(models);
    }

    public static void trainNewModel() throws IOException {
        // First, construct a valid neural network
        NeuralNetwork network = new NeuralNetwork(new Softmax());

        ActivationFunction func = new ReLU();

        network.addHiddenLayer(16, 784, func);
        network.addHiddenLayer(16, 16, func);
        network.addOutputLayer(10, 16);

        // Create a model using the specified network
        Model model = new Model("Teeny", network);

        // Tunable training with epoch count, batch size and learning rate
        model.train(TRAIN_LABEL_FILE,
                TRAIN_IMAGE_FILE,
                LEARNING_RATE, EPOCHS,
                TRAINING_BATCH_SIZE,
                FLASH);

        // Saving the desired model
        model.save();

        model.test(TEST_LABEL_FILE,
                TEST_IMAGE_FILE,
                TESTING_BATCH_SIZE,
                FLASH);
    }

    public static void trainNewModelWithSpecifiedParams(String modelName, NeuralNetwork network, int epochs, double learningRate) throws IOException {
        Model model = new Model(modelName, network);

        model.train(TRAIN_LABEL_FILE,
                TRAIN_IMAGE_FILE,
                learningRate, epochs,
                TRAINING_BATCH_SIZE,
                FLASH);

        model.save();

        model.test(TEST_LABEL_FILE,
                TEST_IMAGE_FILE,
                TESTING_BATCH_SIZE,
                FLASH);
    }

    /*
     * Running the specified models on custom png dataset(s)
     * Note: you can use Microsoft's paint, reduce the canvas size
     * to 28 x 28, make the background pure black with the bucket
     * tool, then uses pure white brushes to draw a handwritten
     * digit. After that, save it as a png and put it in
     * src/mnist/data/png, eg: digit representing 0 should be saved
     * as test0.png, 9 should be saved as test9.png
     */
    public static void testModelsFromPng(List<Model> models) {
        for (Model model : models) {
            System.out.println(model);
            int positive = 0;
            for (int i = 0; i < 10; i++) {
                String expected = String.format("Expected: %d,", i);
                System.out.print(expected);

                int result = model.predict("src/mnist/data/png/test" + i + ".png");

                String positiveIndicator = (i == result ? "O" : "X");
                System.out.printf(", Positive: %s\n", positiveIndicator);

                if (i == result) positive++;
            }
            System.out.printf("Overall: %d / 10%n", positive);
            System.out.println("***\n");
        }
    }


    // Running the specified models on the testing MNIST dataset
    public static void testModelsOnDataSet(List<Model> models) throws IOException {
        for (Model model : models) {
            model.test(TEST_LABEL_FILE,
                    TEST_IMAGE_FILE,
                    TESTING_BATCH_SIZE,
                    FLASH);
            System.out.println("***\n");
        }
    }

    public static void legacyTrainingSession() throws IOException {
        ActivationFunction sigmoid = new Sigmoid();
        ActivationFunction relu = new ReLU();
        ActivationFunction tanh = new Tanh();

        NeuralNetwork network1 = new NeuralNetwork(new Softmax());

        network1.addHiddenLayer(128, 784, sigmoid);
        network1.addOutputLayer(10, 128);

        trainNewModelWithSpecifiedParams("Primus", network1, 10, 0.001);

        NeuralNetwork network2 = new NeuralNetwork(new Softmax());

        network2.addHiddenLayer(128, 784, relu);
        network2.addOutputLayer(10, 128);

        trainNewModelWithSpecifiedParams("PReLUs", network2, 10, 0.0005);

        NeuralNetwork network3 = new NeuralNetwork(new Softmax());

        network3.addHiddenLayer(128, 784, sigmoid);
        network3.addHiddenLayer(64, 128, sigmoid);
        network3.addOutputLayer(10, 64);

        trainNewModelWithSpecifiedParams("Strata", network3, 10, 0.001);

        NeuralNetwork network4 = new NeuralNetwork(new Softmax());

        network4.addHiddenLayer(128, 784, relu);
        network4.addHiddenLayer(64, 128, relu);
        network4.addOutputLayer(10, 64);

        trainNewModelWithSpecifiedParams("StraRe", network4, 5, 0.0005);

        NeuralNetwork network5 = new NeuralNetwork(new Softmax());

        network5.addHiddenLayer(128, 784, relu);
        network5.addHiddenLayer(64, 128, relu);
        network5.addHiddenLayer(32, 64, relu);
        network5.addOutputLayer(10, 32);

        trainNewModelWithSpecifiedParams("Nexus", network5, 5, 0.0005);

        NeuralNetwork network6 = new NeuralNetwork(new Softmax());

        network6.addHiddenLayer(128, 784, tanh);
        network6.addOutputLayer(10, 128);

        trainNewModelWithSpecifiedParams("Prtanh", network6, 5, 0.001);
    }

}
