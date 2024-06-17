package mnist.utility;

/**
 * The ReLU class implements the ActivationFunction interface,
 * providing the ReLU activation function and its derivative.
 * Should be used as activation function of a hidden layer.
 */
public class ReLU implements ActivationFunction {

    /**
     * {@inheritDoc}
     */
    @Override
    public double apply(double x) {
        return Math.max(0, x);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double derivative(double z) {
        return z < 0 ? 0 : 1;
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
        return "ReLU(x)";
    }

}
