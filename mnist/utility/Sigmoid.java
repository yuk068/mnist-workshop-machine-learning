package mnist.utility;

/**
 * The Sigmoid class implements the ActivationFunction interface,
 * providing the sigmoid activation function and its derivative.
 * Should be used as activation function of a hidden layer.
 */
public class Sigmoid implements ActivationFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public double apply(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double z) {
        double sigmoid = apply(z);
        return sigmoid * (1 - sigmoid);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] apply(double[] z) {
        double[] result = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            result[i] = apply(z[i]);
        }
        return result;
    }

    @Override
    public String toString() {
        return "sigmoid(x)";
    }

}
