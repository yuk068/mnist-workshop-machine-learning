package mnist.utility;

import java.util.Arrays;

/**
 * The Softmax class implements the ActivationFunction interface,
 * providing the softmax activation function.
 * Should be used as activation function of the output layer.
 */
public class Softmax implements ActivationFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public double apply(double z) {
        throw new UnsupportedOperationException(this + " does not support such operation.");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double z) {
        throw new UnsupportedOperationException(this + " does not support such operation.");
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] apply(double[] z) {
        double maxLogit = Arrays.stream(z).max().orElse(0.0);
        double sum = 0.0;
        double[] yHat = new double[z.length];

        for (int i = 0; i < z.length; i++) {
            yHat[i] = Math.exp(z[i] - maxLogit);
            sum += yHat[i];
        }

        for (int i = 0; i < z.length; i++) {
            yHat[i] /= sum;
        }

        return yHat;
    }

    @Override
    public String toString() {
        return "softmax(x)";
    }

}
