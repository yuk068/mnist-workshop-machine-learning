package mnist.viewer;

import mnist.utility.MNISTReader;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;

/**
 * The MNISTViewer class is a graphical user interface (GUI) application
 * that displays MNIST dataset images along with their corresponding labels.
 * It uses MNISTReader to fetch images and labels from the dataset files.
 * Images are displayed in a grid layout, and labels are shown below each image.
 * The viewer automatically cycles through images with a delay, providing a
 * dynamic display of the dataset. The main purpose of this class is to confirm
 * that the data was accurately read.
 */
public class MNISTViewer extends JFrame {

    private static final int IMAGE_COUNT = MNISTReader.TESTING_BATCH_SIZE;
    private static final int GRID_ROWS = 4;
    private static final int GRID_COLS = 5;
    private static final int DELAY_MS = 1000;

    private int[][][] images;
    private int[] labels;
    private int currentIndex = 0;
    private MNISTImagePanel[] imagePanels;
    private JLabel[] labelDisplays;

    final boolean invertColor = true;

    /**
     * Constructs an MNISTViewer GUI application.
     * Initializes the frame, sets up layout, reads MNIST dataset files,
     * and populates the grid with images and their corresponding labels.
     * Starts a timer to periodically update images in the viewer.
     */
    public MNISTViewer() {
        setTitle("MNIST Image Viewer");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new GridLayout(GRID_ROWS + 1, GRID_COLS));

        try {
            System.out.println("Reading dataset(s)... this might take awhile...");
            images = MNISTReader.readImages(MNISTReader.TEST_IMAGE_FILE, IMAGE_COUNT, true);
            labels = MNISTReader.readLabels(MNISTReader.TEST_LABEL_FILE, IMAGE_COUNT, true);
        } catch (IOException e) {
            System.out.println("Something went wrong reading dataset(s).");
            return;
        }

        imagePanels = new MNISTImagePanel[GRID_ROWS * GRID_COLS];
        labelDisplays = new JLabel[GRID_ROWS * GRID_COLS];

        for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) {
            imagePanels[i] = new MNISTImagePanel(images[i], invertColor);
            labelDisplays[i] = new JLabel("Label: " + labels[i], SwingConstants.CENTER);
            labelDisplays[i].setOpaque(true);
            labelDisplays[i].setBackground(Color.BLACK);
            labelDisplays[i].setForeground(Color.CYAN);
            add(imagePanels[i]);
            add(labelDisplays[i]);
        }

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        Timer timer = new Timer(DELAY_MS, e -> updateImages());
        timer.start();
    }

    /**
     * Updates the images and labels displayed in the viewer.
     * Shifts the current index to show the next set of images and labels in the dataset.
     * Called periodically by a timer to provide a slideshow effect of the dataset.
     */
    private void updateImages() {
        for (int i = 0; i < GRID_ROWS * GRID_COLS; i++) {
            int index = (currentIndex + i) % IMAGE_COUNT;
            imagePanels[i].setImage(images[index]);
            labelDisplays[i].setText("Label: " + labels[index]);
        }
        currentIndex = (currentIndex + GRID_ROWS * GRID_COLS) % IMAGE_COUNT;
    }

    /**
     * The main method to start the MNISTViewer application.
     * Creates an instance of MNISTViewer, which initializes and displays the GUI.
     *
     * @param args the command-line arguments (not used)
     */
    public static void main(String[] args) {
        new MNISTViewer();
    }

}
