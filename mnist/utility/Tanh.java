package mnist.utility;

/**
 * The Tanh class implements the ActivationFunction interface,
 * providing the tanh activation function and its derivative.
 * Should be used as activation function of a hidden layer.
 */
public class Tanh implements ActivationFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public double apply(double z) {
        return Math.tanh(z);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double z) {
        double tanh = Math.tanh(z);
        return 1 - tanh * tanh;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] apply(double[] z) {
        throw new UnsupportedOperationException(this + " does not support such operation.");
    }

    @Override
    public String toString() {
        return "tanh(x)";
    }

}
