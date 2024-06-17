# MNIST Workshop - Yuk

This project represents a fundamental exploration into the realm of machine learning using Java, focusing on the MNIST dataset—a classic benchmark in the field. Serving as an introductory foray into machine learning, the interface encapsulates essential functionalities tailored towards MNIST models. Users can initiate the creation of neural network models, train them with specified parameters, conduct testing to evaluate performance metrics, utilize trained models for predictions, and finally, save these models for future use. By leveraging Java's core functionalities without dependencies on additional packages, this project emphasizes foundational concepts such as model creation, training optimization through parameter adjustments like learning rate and epochs, validation via testing procedures, and the application of trained models for predictive tasks.

# User guide:

Move package `/mnist` to your `src` directory of your project in your Java IDE. Then, you can review the source code, and start testing the program in `mnist/training/MNISTWorkshop`. Thank you for trying out my program!

# Project structure and classes overview:

## I, mnist/data

- Stores the original MNIST datasets, which can be found on [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).
- Contains a folder /png to store custom MNIST images.

## II, mnist/utility

### Class: MNISTReader

The `MNISTReader` class provides utility methods for reading and handling MNIST dataset files, offering functionalities for loading and saving data efficiently.

#### Key Features:

1. **Label and Image Reading**:
    - **Method**: `readLabelsFromDisk(String labelFilePath, int batchSize)`
    - **Description**: Reads MNIST labels from a specified file path.
    - **Method**: `readImagesFromDisk(String imageFilePath, int batchSize)`
    - **Description**: Reads MNIST images from a specified file path.

2. **Caching Mechanism**:
    - **Method**: `serializeLabels(int[] labels)`
    - **Description**: Serializes and saves MNIST labels to a cache file.
    - **Method**: `deserializeLabels()`
    - **Description**: Deserializes MNIST labels from a cache file.
    - **Method**: `serializeImages(int[][][] images)`
    - **Description**: Serializes and saves MNIST images to a cache file.
    - **Method**: `deserializeImages()`
    - **Description**: Deserializes MNIST images from a cache file.

3. **Image Loading**:
    - **Method**: `loadImage(String filePath)`
    - **Description**: Loads an image from a specified file path and converts it to grayscale pixels.

The `MNISTReader` class ensures efficient handling of MNIST data through direct file operations and caching mechanisms, optimizing the workflow for training and testing neural networks.

### Interface: ActivationFunction

The `ActivationFunction` interface provides a structure for implementing various activation functions used in neural networks. It is designed to be utilized for both hidden layers and output layers, ensuring flexibility and consistency across different parts of the network. The interface extends `Serializable`, allowing implementing classes to be serialized and enabling model persistence.

#### Key Features:

1. **Single Input Application**:
    - **Method**: `apply(double x)`
    - **Description**: Applies the activation function to a single input value.
    - **Exception**: Throws `UnsupportedOperationException` if the function doesn't support single variable application.

2. **Derivative Calculation**:
    - **Method**: `derivative(double z)`
    - **Description**: Computes the derivative of the activation function given the pre-activation input, useful for backpropagation during training.
    - **Exception**: Throws `UnsupportedOperationException` if the function doesn't support single variable derivative calculation.

3. **Array Input Application**:
    - **Method**: `apply(double[] z)`
    - **Description**: Applies the activation function to each element of an input array, typically used for output layers.
    - **Exception**: Throws `UnsupportedOperationException` if the function doesn't support array application.

The package includes implementations for the following activation functions:
- **Softmax**
- **Sigmoid**
- **ReLU (Rectified Linear Unit)**
- **Tanh (Hyperbolic Tangent)**

Implementing classes are expected to provide appropriate logic for these operations, catering to the specific characteristics of each activation function. If an operation is not applicable for a specific activation function, the implementing class should handle it by throwing an `UnsupportedOperationException`. This design ensures that each activation function can be used effectively and safely within neural network architectures.

## III, mnist/cache

- This directory stores cached data from `MNISTReader` to enable quick full-length reading and large batch-size reading.
- Cached labels and images are saved and loaded from here to optimize data processing speed during training and testing.

## IV, mnist/model

- This directory is dedicated to storing saved models.
- Trained models are saved here and can be loaded for future use.
- Contains a few pretrained models for immediate deployment and testing.

## V, mnist/viewer

### Class: MNISTImagePanel

The `MNISTImagePanel` class represents a panel in a graphical user interface (GUI) that displays an MNIST image. It supports displaying an image matrix and allows inverting colors optionally. This panel is used with the `MNISTViewer` class to visualize MNIST dataset images by decomposing and recomposing them.

#### Key Features:

1. **Image Display**:
    - **Constructor**: `MNISTImagePanel(int[][] image, boolean invertColors)`
    - **Description**: Initializes the panel with an image matrix and an optional color inversion setting.
    - **Method**: `setImage(int[][] image)`
    - **Description**: Updates the image matrix displayed on the panel.

2. **Custom Painting**:
    - **Method**: `paintComponent(Graphics g)`
    - **Description**: Overrides the paint method to render the MNIST image on the panel.

### Class: MNISTViewer

The `MNISTViewer` class is a graphical user interface (GUI) application that displays MNIST dataset images along with their corresponding labels. It uses `MNISTReader` to fetch images and labels from the dataset files. Images are displayed in a grid layout, and labels are shown below each image. The viewer automatically cycles through images with a delay, providing a dynamic display of the dataset. The main purpose of this class is to confirm that the data was accurately read.

#### Key Features:

1. **GUI Initialization**:
    - **Constructor**: `MNISTViewer()`
    - **Description**: Initializes the frame, sets up the layout, reads MNIST dataset files, and populates the grid with images and their corresponding labels. Starts a timer to periodically update images in the viewer.

2. **Dynamic Image Display**:
    - **Method**: `updateImages()`
    - **Description**: Updates the images and labels displayed in the viewer. Shifts the current index to show the next set of images and labels in the dataset, providing a slideshow effect.

3. **Main Method**:
    - **Method**: `main(String[] args)`
    - **Description**: The entry point to start the MNISTViewer application. Creates an instance of `MNISTViewer`, which initializes and displays the GUI.

## VI. mnist/training

The `mnist/training` package is dedicated to the core components involved in training neural networks. This package includes classes representing the fundamental building blocks of a neural network, such as neurons, layers, and the network itself. Below are the brief descriptions of the key classes in this package:

### Class: Neuron

The `Neuron` class represents a single neuron in a neural network. It includes the neuron's weights, bias, activation function, and methods for performing forward propagation and parameter updates.

#### Key Features:

1. **Initialization**:
   - **Constructor**: `Neuron(int inputSize, ActivationFunction function)`
   - **Description**: Constructs a neuron with the specified number of inputs and activation function. Initializes weights and bias with random values.

2. **Forward Propagation**:
   - **Method**: `forward(double[] inputs)`
   - **Description**: Performs forward propagation by calculating the weighted sum of inputs, adding the bias, and applying the activation function.

3. **Parameter Updates**:
   - **Method**: `updateParams(double[] gradients, double learningRate)`
   - **Description**: Updates the weights and bias of the neuron using the provided gradients and learning rate.

### Class: NeuronLayer

The `NeuronLayer` class represents a layer of neurons in a neural network. It encapsulates multiple neurons, an activation function, and methods for performing forward propagation and parameter updates across the layer.

#### Key Features:

1. **Initialization**:
   - **Constructor**: `NeuronLayer(int numNeurons, int inputSize, ActivationFunction activation, boolean lastLayer)`
   - **Description**: Constructs a layer of neurons with a specified number of neurons, each having a specified number of inputs, an activation function, and a flag indicating if it is the last layer in the network.

2. **Forward Propagation**:
   - **Method**: `forward(double[] inputs)`
   - **Description**: Propagates input values through the layer, computing outputs of each neuron. If it is the last layer, applies the activation function to compute output probabilities.

3. **Parameter Updates**:
   - **Method**: `updateParams(double[][] gradients, double learningRate)`
   - **Description**: Updates weights and biases of all neurons in the layer using gradients and a specified learning rate.

This class serves as a fundamental component for organizing and processing data through neural networks, facilitating training and inference operations efficiently.

### Class: NeuralNetwork

The `NeuralNetwork` class represents a neural network composed of layers of neurons. It facilitates the construction, training, testing, and prediction functionalities of the network.

#### Key Features:

1. **Initialization**:
   - **Constructor**: `NeuralNetwork(ActivationFunction outputActivation)`
   - **Description**: Constructs a neural network with a specified output activation function.

2. **Adding Layers**:
   - **Method**: `addHiddenLayer(int numNeurons, int inputSize, ActivationFunction activation)`
   - **Description**: Adds a hidden layer with a specified number of neurons, input size, and activation function.
   - **Method**: `addOutputLayer(int outputClassificationSize, int inputSize)`
   - **Description**: Adds an output layer with a specified number of neurons and input size, using the output activation function specified during initialization.

3. **Forward Propagation**:
   - **Method**: `forwardPass(double[] inputs, int layerIndex)`
   - **Description**: Performs a forward pass through the network up to the specified layer index, computing output values.

4. **Backward Propagation**:
   - **Method**: `backwardPass(double[] target, double[] predicted)`
   - **Description**: Computes the deltas (errors) for each layer during backpropagation based on the target and predicted output values.

5. **Calculating Gradients**:
   - **Method**: `calculateGradients(double[] inputs, double[][] deltas)`
   - **Description**: Calculates gradients for all layers based on the inputs and deltas computed during backpropagation.

6. **Updating Parameters**:
   - **Method**: `updateParams(double[][][] gradients, double learningRate)`
   - **Description**: Updates the weights and biases of all layers using the calculated gradients and specified learning rate.

7. **Training**:
   - **Method**: `train(String labelFilePath, String imageFilePath, double learningRate, int epochs, int batchSize, boolean flashRead)`
   - **Description**: Trains the neural network using the MNIST dataset, specifying label and image file paths, learning rate, number of epochs, batch size, and whether to use flash read for dataset processing.

8. **Testing**:
   - **Method**: `test(String labelFilePath, String imageFilePath, int batchSize, boolean flashRead)`
   - **Description**: Tests the neural network's accuracy using the MNIST dataset, specifying label and image file paths, batch size, and whether to use flash read for dataset processing.

9. **Prediction**:
   - **Method**: `predict(double[] inputs)`
   - **Description**: Predicts the label of input values using the neural network.
   - **Method**: `predict(String imagePath)`
   - **Description**: Predicts the label and confidence of an image specified by the file path.

This class serves as the central component for building, training, and using neural networks for classification tasks, leveraging the flexibility and power of multi-layered networks for complex pattern recognition.

### Class: Model

The `Model` class represents a serializable neural network model designed for training, testing, predicting, saving, and loading MNIST data.

#### Key Features:

1. **Initialization**:
   - **Constructor**: `Model(String name, NeuralNetwork neuralNetwork)`
   - **Description**: Constructs a model with a specified name and associated neural network. It validates the network configuration to ensure it meets required specifications.

2. **Training**:
   - **Method**: `train(String labelFilePath, String imageFilePath, double learningRate, int epochs, int batchSize, boolean flashRead)`
   - **Description**: Trains the model using the MNIST dataset, specifying label and image file paths, learning rate, number of epochs, batch size, and whether to use flash read for dataset processing.

3. **Testing**:
   - **Method**: `test(String labelFilePath, String imageFilePath, int batchSize, boolean flashRead)`
   - **Description**: Tests the model's accuracy using the MNIST dataset, specifying label and image file paths, batch size, and whether to use flash read for dataset processing.

4. **Prediction**:
   - **Method**: `predict(String imagePath)`
   - **Description**: Predicts the label and confidence for an image specified by the file path using the associated neural network.

5. **Serialization**:
   - **Method**: `save()`
   - **Description**: Saves the model to a file using Java serialization.

6. **Deserialization**:
   - **Method**: `load(String modelName)`
   - **Description**: Loads a serialized model from a file based on the given model name.

7. **Validation**:
   - **Method**: `validateNetwork()`
   - **Description**: Ensures the neural network configuration meets required specifications:
     - At least one output layer, which must be the last layer.
     - At least one hidden layer.
     - Network must have at least two layers.
     - Input layer must have 784 neurons corresponding to MNIST image size.
     - Subsequent layers must match in size with the previous layer.

8. **String Representation**:
   - **Method**: `toString()`
   - **Description**: Returns a string representation of the model including its name, network parameters (weights and biases), and details of each layer in the neural network.

This class encapsulates the functionality to effectively manage, train, and utilize a neural network model for MNIST digit classification tasks, offering methods for training, testing, prediction, serialization, and deserialization.

## Personal note:

Despite recognizing that Java may not be the most efficient language for machine learning tasks like the MNIST dataset, I believe that developing this project from scratch without relying on external packages demonstrates a thorough grasp of core algorithms and underlying machine learning principles. This project serves as a foundational exploration into the realm of machine learning, focusing specifically on the MNIST dataset—a longstanding benchmark in the field.

The project interface provides essential functionalities tailored for MNIST models. Users can initiate the creation of neural network models, customize training parameters such as learning rate and epochs, evaluate model performance through testing, utilize trained models for predictions, and save them for future use. By leveraging Java’s core functionalities and abstaining from external dependencies, the project emphasizes fundamental concepts including model creation, optimization, validation, and application in predictive tasks.

I acknowledge that this implementation has limitations and cannot match the capabilities of popular machine learning frameworks such as TensorFlow or PyTorch. Nonetheless, I am proud of the knowledge I gained and achievements made through this project. I hope that you will also find value in this endeavor, should you choose to further review the source code.

Thank you for reading and trying out my program!
