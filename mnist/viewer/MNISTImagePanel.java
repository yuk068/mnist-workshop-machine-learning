package mnist.viewer;

import javax.swing.*;
import java.awt.*;

/**
 * The MNISTImagePanel class represents a panel in a graphical user interface (GUI)
 * that displays an MNIST image. It supports displaying an image matrix and allows
 * inverting colors optionally. This panel is used with the MNISTViewer
 * class to visualize MNIST dataset images by decomposing and recomposing them.
 */
public class MNISTImagePanel extends JPanel {

    private int[][] image;
    private static final int PIXEL_SIZE = 5;
    private final boolean invertColors;

    /**
     * Constructs an MNISTImagePanel with the specified image matrix and color inversion setting.
     *
     * @param image        the 2D array representing the image pixels (grayscale values)
     * @param invertColors true to invert colors (white becomes black and vice versa), false otherwise
     */
    public MNISTImagePanel(int[][] image, boolean invertColors) {
        this.image = image;
        this.invertColors = invertColors;
        this.setPreferredSize(new Dimension(image[0].length * PIXEL_SIZE, image.length * PIXEL_SIZE));
    }

    /**
     * Sets a new image matrix to be displayed on the panel.
     *
     * @param image the new 2D array representing the image pixels (grayscale values)
     */
    public void setImage(int[][] image) {
        this.image = image;
        repaint();
    }

    /**
     * Overrides the paintComponent method to paint the image on the panel.
     *
     * @param g the Graphics object used for painting
     */
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        for (int i = 0; i < image.length; i++) {
            for (int j = 0; j < image[i].length; j++) {
                int colorValue = invertColors ? image[i][j] : 255 - image[i][j];
                g.setColor(new Color(colorValue, colorValue, colorValue));
                g.fillRect(j * PIXEL_SIZE, i * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
            }
        }
    }

}
