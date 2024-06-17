package mnist.utility;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.Arrays;

/**
 * The MNISTReader class provides utility methods to read MNIST dataset files
 * and perform operations related to loading and saving data.
 */
public class MNISTReader {

    public static final int LABEL_MAGIC_NUMBER = 2049;
    public static final int IMAGE_MAGIC_NUMBER = 2051;
    public static final int TRAINING_BATCH_SIZE = 60000;
    public static final int TESTING_BATCH_SIZE = 10000;

    public static final String TRAIN_LABEL_FILE = "src/mnist/data/train-labels.idx1-ubyte";
    public static final String TRAIN_IMAGE_FILE = "src/mnist/data/train-images.idx3-ubyte";
    public static final String TEST_LABEL_FILE = "src/mnist/data/t10k-labels.idx1-ubyte";
    public static final String TEST_IMAGE_FILE = "src/mnist/data/t10k-images.idx3-ubyte";

    public static final String LABEL_CACHE_FILE = "src/mnist/cache/mnist_labels.ser";
    public static final String IMAGE_CACHE_FILE = "src/mnist/cache/mnist_images.ser";

    /**
     * Reads MNIST labels from the disk.
     *
     * @param labelFilePath the file path to the MNIST label data
     * @param batchSize     the number of labels to read
     * @return an array of labels read from the file
     * @throws IOException if an I/O error occurs or if the label file is invalid
     */
    public static int[] readLabelsFromDisk(String labelFilePath, int batchSize) throws IOException {
        try (DataInputStream labelStream = new DataInputStream(new FileInputStream(labelFilePath))) {
            int magicNumber = labelStream.readInt();
            if (magicNumber != LABEL_MAGIC_NUMBER) {
                throw new IOException("Invalid MNIST label file.");
            }
            int numberOfLabels = labelStream.readInt();
            int[] labels = new int[Math.min(numberOfLabels, batchSize)];
            for (int i = 0; i < labels.length; i++) {
                labels[i] = labelStream.readUnsignedByte();
            }
            return labels;
        }
    }

    /**
     * Reads MNIST images from the disk.
     *
     * @param imageFilePath the file path to the MNIST image data
     * @param batchSize     the number of images to read
     * @return a 3D array representing the images read from the file
     * @throws IOException if an I/O error occurs or if the image file is invalid
     */
    public static int[][][] readImagesFromDisk(String imageFilePath, int batchSize) throws IOException {
        try (DataInputStream imageStream = new DataInputStream(new FileInputStream(imageFilePath))) {
            int magicNumber = imageStream.readInt();
            if (magicNumber != IMAGE_MAGIC_NUMBER) {
                throw new IOException("Invalid MNIST image file.");
            }
            int numberOfImages = imageStream.readInt();
            int numRows = imageStream.readInt();
            int numCols = imageStream.readInt();
            int[][][] images = new int[Math.min(numberOfImages, batchSize)][][];
            for (int i = 0; i < images.length; i++) {
                images[i] = new int[numRows][numCols];
                for (int r = 0; r < numRows; r++) {
                    for (int c = 0; c < numCols; c++) {
                        images[i][r][c] = imageStream.readUnsignedByte();
                    }
                }
            }
            return images;
        }
    }

    /**
     * Reads MNIST labels, either from cache or from disk.
     *
     * @param labelFilePath the file path to the MNIST label data
     * @param batchSize     the number of labels to read
     * @return an array of labels read from the file
     * @throws IOException if an I/O error occurs or if the label file is invalid
     */
    public static int[] readLabels(String labelFilePath, int batchSize) throws IOException {
        int[] cachedLabels = deserializeLabels();
        if (cachedLabels != null) {
            return Arrays.copyOf(cachedLabels, batchSize);
        }

        int[] labels = readLabelsFromDisk(labelFilePath, batchSize);
        serializeLabels(labels);

        return Arrays.copyOf(labels, batchSize);
    }

    /**
     * Reads MNIST images, either from cache or from disk.
     *
     * @param imageFilePath the file path to the MNIST image data
     * @param batchSize     the number of images to read
     * @return a 3D array representing the images read from the file
     * @throws IOException if an I/O error occurs or if the image file is invalid
     */
    public static int[][][] readImages(String imageFilePath, int batchSize) throws IOException {
        int[][][] cachedImages = deserializeImages();
        if (cachedImages != null) {
            return Arrays.copyOf(cachedImages, batchSize);
        }

        int[][][] images = readImagesFromDisk(imageFilePath, batchSize);
        serializeImages(images);

        return Arrays.copyOf(images, batchSize);
    }

    /**
     * Serializes MNIST labels to a cache file.
     *
     * @param labels the array of labels to serialize
     */
    public static void serializeLabels(int[] labels) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(LABEL_CACHE_FILE))) {
            oos.writeObject(labels);
        } catch (IOException e) {
            System.out.println("Serialization of labels failed.");
        }
    }

    /**
     * Deserializes MNIST labels from a cache file.
     *
     * @return the array of labels deserialized from the cache file, or null if deserialization fails
     */
    public static int[] deserializeLabels() {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(LABEL_CACHE_FILE))) {
            return (int[]) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            return null;
        }
    }

    /**
     * Serializes MNIST images to a cache file.
     *
     * @param images the 3D array of images to serialize
     */
    public static void serializeImages(int[][][] images) {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(IMAGE_CACHE_FILE))) {
            oos.writeObject(images);
        } catch (IOException e) {
            System.out.println("Serialization of images failed.");
        }
    }

    /**
     * Deserializes MNIST images from a cache file.
     *
     * @return the 3D array of images deserialized from the cache file, or null if deserialization fails
     */
    public static int[][][] deserializeImages() {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(IMAGE_CACHE_FILE))) {
            return (int[][][]) ois.readObject();
        } catch (IOException | ClassNotFoundException e) {
            return null;
        }
    }

    /**
     * Reads MNIST labels from disk without caching.
     *
     * @param labelFilePath the file path to the MNIST label data
     * @param batchSize     the number of labels to read
     * @return an array of labels read from the file
     * @throws IOException if an I/O error occurs or if the label file is invalid
     */
    public static int[] flashReadLabels(String labelFilePath, int batchSize) throws IOException {
        return readLabelsFromDisk(labelFilePath, batchSize);
    }

    /**
     * Reads MNIST images from disk without caching.
     *
     * @param imageFilePath the file path to the MNIST image data
     * @param batchSize     the number of images to read
     * @return a 3D array representing the images read from the file
     * @throws IOException if an I/O error occurs or if the image file is invalid
     */
    public static int[][][] flashReadImages(String imageFilePath, int batchSize) throws IOException {
        return readImagesFromDisk(imageFilePath, batchSize);
    }

    /**
     * Loads an image from the specified file path and converts it to grayscale pixels.
     *
     * @param filePath the file path to the image
     * @return a 2D array of grayscale pixel values representing the image
     */
    public static int[][] loadImage(String filePath) {
        BufferedImage img = null;
        try {
            img = ImageIO.read(new File(filePath));
        } catch (IOException e) {
            System.out.println("Image not found.");
        }
        assert img != null;
        int width = img.getWidth();
        int height = img.getHeight();
        int[][] pixels = new int[width][height];

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                int color = img.getRGB(x, y);
                int red = (color >> 16) & 0xff;
                int green = (color >> 8) & 0xff;
                int blue = color & 0xff;
                int gray = (red + green + blue) / 3;
                pixels[x][y] = gray;
            }
        }

        return pixels;
    }

}


