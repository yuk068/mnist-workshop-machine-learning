package mnist.utility;

import java.io.Serializable;

/**
 * The ActivationFunction interface provides methods for applying an activation function
 * and its derivative in neural networks. This interface extends Serializable, enabling
 * implementing classes to be serialized.
 * <p>
 * This interface is designed to support activation functions for both hidden layers and
 * output layers in neural networks. Implementing classes should provide appropriate
 * operations for each type of activation function.
 * <p>
 * If an operation is not supported for a specific activation function (e.g., applying to
 * an array for some activation functions), implementing classes should throw an
 * UnsupportedOperationException.
 */
public interface ActivationFunction extends Serializable {

    /**
     * Applies the activation function to a single input value.
     *
     * @param x the input value
     * @return the result of applying the activation function
     * @throws UnsupportedOperationException if the activation function does not support
     *                                       applying a single variable
     */
    double apply(double x);

    /**
     * Computes the derivative of the activation function given the pre-activation input.
     *
     * @param z the pre-activation input
     * @return the derivative of the activation function
     * @throws UnsupportedOperationException if the activation function does not support
     *                                       taking derivative of a single variable
     */
    double derivative(double z);

    /**
     * Applies the activation function to each element of the input array.
     * Usually used for output activation functions.
     *
     * @param z the input array
     * @return an array containing the results of applying the activation function
     * @throws UnsupportedOperationException if the activation function does not support
     *                                       applying to arrays
     */
    double[] apply(double[] z);

}
