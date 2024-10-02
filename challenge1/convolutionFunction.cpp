#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm> // For std::min and std::max
#include <vector>    // For std::vector
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <random>

using namespace Eigen;
using namespace std;
// Funzione per costruire la matrice di convoluzione
SparseMatrix<double> buildSparseConvolutionMatrix(const MatrixXd& kernel, int imageHeight, int imageWidth) {
    int kernelRows = kernel.rows();
    int kernelCols = kernel.cols();
    int imageSize = imageHeight * imageWidth;
    std::vector<Triplet<double>> triplets;

    // Estima il numero massimo di elementi non-zero per pre-allocare lo spazio
    triplets.reserve(imageSize * kernelRows * kernelCols);

    // Popolare il vettore dei triplet
    for (int i = 0; i < imageHeight; ++i) {
        for (int j = 0; j < imageWidth; ++j) {
            int rowIndex = i * imageWidth + j;
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    int ii = i + ki - kernelRows / 2;
                    int jj = j + kj - kernelCols / 2;
                    if (ii >= 0 && ii < imageHeight && jj >= 0 && jj < imageWidth) {
                        int colIndex = ii * imageWidth + jj;
                        double value = kernel(ki, kj);
                        // Considera solo i valori non nulli
                        if (value != 0.0) {
                            triplets.push_back(Triplet<double>(rowIndex, colIndex, value));
                        }
                    }
                }
            }
        }
    }

    // Costruire la matrice sparsa dai triplet
    SparseMatrix<double> convolutionMatrix(imageSize, imageSize);
    convolutionMatrix.setFromTriplets(triplets.begin(), triplets.end());
    
    return convolutionMatrix;
}

    
    

// Function to apply 2D convolution with boundary handling
MatrixXd applyConvolution(const MatrixXd& input, const MatrixXd& kernel) {
    int inputRows = input.rows();
    int inputCols = input.cols();
    int kernelRows = kernel.rows();
    int kernelCols = kernel.cols();
    
    // Output dimensions
    int outputRows = inputRows;
    int outputCols = inputCols;
    
    // Initialize the output matrix
    MatrixXd output(outputRows, outputCols);
    
    // Perform the convolution
    for (int i = 0; i < outputRows; ++i) {
        for (int j = 0; j < outputCols; ++j) {
            double value = 0.0;
            for (int ki = 0; ki < kernelRows; ++ki) {
                for (int kj = 0; kj < kernelCols; ++kj) {
                    int ii = i + ki - kernelRows / 2;
                    int jj = j + kj - kernelCols / 2;
                    if (ii >= 0 && ii < inputRows && jj >= 0 && jj < inputCols) {
                        value += input(ii, jj) * kernel(ki, kj);
                    }
                }
            }
            // Clamp the value between 0 and 255
            value = std::max(0.0, std::min(255.0, value)); // Ensure both arguments are of type double
            output(i, j) = value;
        }
    }
    
    return output;
}
// Function to add random noise to an image
MatrixXd addRandomNoise(const MatrixXd& input, int minNoise, int maxNoise) {
    int rows = input.rows();
    int cols = input.cols();
    MatrixXd noisyImage(rows, cols);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(minNoise, maxNoise);

    // Add noise to each pixel
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double noise = dis(gen);
            noisyImage(i, j) = std::max(0.0, std::min(255.0, input(i, j) + noise));
        }
    }

    return noisyImage;
}

int main() {
    // Path to the image
    const char* input_image_path = "Einstein.jpg";

    // Load the image using stb_image in grayscale
    int width, height, channels;
    unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);  // Force load as grayscale
    if (!image_data) {
        std::cerr << "Error: Could not load image " << input_image_path << std::endl;
        return 1;
    }

    std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;

    // Prepare Eigen matrix for the grayscale image
    MatrixXd image(height, width);

    // Fill the matrix with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            image(i, j) = static_cast<double>(image_data[i * width + j]);
        }
    }

    // Free memory
    stbi_image_free(image_data);

    // Define kernels as matrices
    MatrixXd smoothing(3, 3);
    smoothing << 0, 1, 0,
                 1, 4, 1,
                 0, 1, 0;
    

    MatrixXd sharpening(3, 3);
    sharpening <<  0, -1,  0,
                  -1,  5, -1,
                   0, -1,  0;

    MatrixXd edge_detection(3, 3);
    edge_detection << -1, -1, -1,
                      -1,  8, -1,
                      -1, -1, -1;

/*// Apply the convolution using the smoothing kernel
MatrixXd smoothed_image = applyConvolution(image, smoothing);

// Apply the convolution using the sharpening kernel
MatrixXd sharpened_image = applyConvolution(image, sharpening);

// Apply the convolution using the edge detection kernel
MatrixXd edge_detected_image = applyConvolution(image, edge_detection);
*/
// Add random noise to the original image
MatrixXd noisy_image = addRandomNoise(image, -50, 50);

    // Convert the Eigen matrices to unsigned char arrays for saving
   // unsigned char* smoothed_image_data = new unsigned char[width * height];
    unsigned char* sharpened_image_data = new unsigned char[width * height];
    unsigned char* edge_detected_image_data = new unsigned char[width * height];
    unsigned char* noisyMatrix = new unsigned char[width * height];

    // Reshape the original and noisy images as vectors
    Eigen::VectorXd v = Eigen::Map<Eigen::VectorXd>(image.data(), image.size());
    Eigen::VectorXd w = Eigen::Map<Eigen::VectorXd>(noisy_image.data(), noisy_image.size());
     // Verify that each vector has m * n components
    std::cout << "Original vector size: " << v.size() << std::endl;
    std::cout << "Noisy vector size: " << w.size() << std::endl;


    // Compute and report the Euclidean norm of the original image vector
    double norm_v = v.norm();
    std::cout << "Euclidean norm of the original image vector: " << norm_v << std::endl;    #include <iostream>
    #include <Eigen/Dense>
    #include <Eigen/Sparse>
    #include <algorithm> // For std::min and std::max
    #include <vector>    // For std::vector
    #include <random>
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
    #define STB_IMAGE_WRITE_IMPLEMENTATION
    #include "stb_image_write.h"
    
    using namespace Eigen;
    using namespace std;
    
    // Funzione per costruire la matrice di convoluzione sparsa
    SparseMatrix<double> buildSparseConvolutionMatrix(const MatrixXd& kernel, int imageHeight, int imageWidth, int& nonZeroCount) {
        int kernelRows = kernel.rows();
        int kernelCols = kernel.cols();
        int imageSize = imageHeight * imageWidth;
        std::vector<Triplet<double>> triplets;
    
        // Pre-allocare il vettore con una stima del numero di elementi non-zero
        triplets.reserve(imageSize * kernelRows * kernelCols);
    
        // Inizializzare il contatore degli elementi non-zero
        nonZeroCount = 0;
    
        // Popolare il vettore dei triplet
        for (int i = 0; i < imageHeight; ++i) {
            for (int j = 0; j < imageWidth; ++j) {
                int rowIndex = i * imageWidth + j;
                for (int ki = 0; ki < kernelRows; ++ki) {
                    for (int kj = 0; kj < kernelCols; ++kj) {
                        int ii = i + ki - kernelRows / 2;
                        int jj = j + kj - kernelCols / 2;
                        if (ii >= 0 && ii < imageHeight && jj >= 0 && jj < imageWidth) {
                            int colIndex = ii * imageWidth + jj;
                            double value = kernel(ki, kj);
                            // Considera solo i valori non nulli
                            if (value != 0.0) {
                                triplets.push_back(Triplet<double>(rowIndex, colIndex, value));
                                nonZeroCount++;
                            }
                        }
                    }
                }
            }
        }
    
        // Costruire la matrice sparsa dai triplet
        SparseMatrix<double> convolutionMatrix(imageSize, imageSize);
        convolutionMatrix.setFromTriplets(triplets.begin(), triplets.end());
    
        return convolutionMatrix;
    }
    
    // Funzione per applicare la convoluzione 2D con gestione dei bordi
    MatrixXd applyConvolution(const MatrixXd& input, const MatrixXd& kernel) {
        int inputRows = input.rows();
        int inputCols = input.cols();
        int kernelRows = kernel.rows();
        int kernelCols = kernel.cols();
    
        // Output dimensions
        int outputRows = inputRows;
        int outputCols = inputCols;
    
        // Initialize the output matrix
        MatrixXd output(outputRows, outputCols);
    
        // Perform the convolution
        for (int i = 0; i < outputRows; ++i) {
            for (int j = 0; j < outputCols; ++j) {
                double value = 0.0;
                for (int ki = 0; ki < kernelRows; ++ki) {
                    for (int kj = 0; kj < kernelCols; ++kj) {
                        int ii = i + ki - kernelRows / 2;
                        int jj = j + kj - kernelCols / 2;
                        if (ii >= 0 && ii < inputRows && jj >= 0 && jj < inputCols) {
                            value += input(ii, jj) * kernel(ki, kj);
                        }
                    }
                }
                // Clamp the value between 0 and 255
                value = std::max(0.0, std::min(255.0, value));
                output(i, j) = value;
            }
        }
    
        return output;
    }
    
    // Funzione per aggiungere rumore casuale a un'immagine
    MatrixXd addRandomNoise(const MatrixXd& input, int minNoise, int maxNoise) {
        int rows = input.rows();
        int cols = input.cols();
        MatrixXd noisyImage(rows, cols);
    
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(minNoise, maxNoise);
    
        // Add noise to each pixel
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double noise = dis(gen);
                noisyImage(i, j) = std::max(0.0, std::min(255.0, input(i, j) + noise));
            }
        }
    
        return noisyImage;
    }
    
    int main() {
        // Path to the image
        const char* input_image_path = "Einstein.jpg";
    
        // Load the image using stb_image in grayscale
        int width, height, channels;
        unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);  // Force load as grayscale
        if (!image_data) {
            std::cerr << "Error: Could not load image " << input_image_path << std::endl;
            return 1;
        }
    
        std::cout << "Image loaded: " << width << "x" << height << " with " << channels << " channels." << std::endl;
    
        // Prepare Eigen matrix for the grayscale image
        MatrixXd image(height, width);
    
        // Fill the matrix with image data
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                image(i, j) = static_cast<double>(image_data[i * width + j]);
            }
        }
    
        // Free memory
        stbi_image_free(image_data);
    
        // Add random noise to the original image
        MatrixXd noisy_image = addRandomNoise(image, -50, 50);
    
        // Reshape the original and noisy images as vectors
        Eigen::VectorXd v = Eigen::Map<Eigen::VectorXd>(image.data(), image.size());
        Eigen::VectorXd w = Eigen::Map<Eigen::VectorXd>(noisy_image.data(), noisy_image.size());
    
        // Verify that each vector has m * n components
        std::cout << "Original vector size: " << v.size() << std::endl;
        std::cout << "Noisy vector size: " << w.size() << std::endl;
    
        // Compute and report the Euclidean norm of the original image vector
        double norm_v = v.norm();
        std::cout << "Euclidean norm of the original image vector: " << norm_v << std::endl;
    
        // Define the smoothing kernel
        MatrixXd smoothing(3, 3);
        smoothing << 1, 1, 1,
                     1, 1, 1,
                     1, 1, 1;
        smoothing /= 9.0; // Normalize the kernel
    
        // Compute the sparse convolution matrix
        int nonZeroCount = 0;
        Eigen::SparseMatrix<double> A1 = buildSparseConvolutionMatrix(smoothing, height, width, nonZeroCount);
    
        // Print the number of non-zero elements
        std::cout << "Number of non-zero elements in the convolution matrix: " << nonZeroCount << std::endl;
    
        // Now create a vector which is the product between the sparse matrix A1 and the vector v
        Eigen::VectorXd Av = A1 * v;
    
        // Convert the result back to a matrix
        Eigen::MatrixXd smoothed_image = Eigen::Map<Eigen::MatrixXd>(Av.data(), height, width);
    
        // Print the norm of the resulting vector
        std::cout << "Norm of the resulting vector: " << Av.norm() << std::endl;
    
        // Convert the Eigen matrices to unsigned char arrays for saving
        unsigned char* smoothed_image_data = new unsigned char[width * height];
        unsigned char* sharpened_image_data = new unsigned char[width * height];
        unsigned char* edge_detected_image_data = new unsigned char[width * height];
        unsigned char* noisyMatrix = new unsigned char[width * height];
    
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                smoothed_image_data[i * width + j] = static_cast<unsigned char>(smoothed_image(i, j));
                sharpened_image_data[i * width + j] = static_cast<unsigned char>(image(i, j));
                edge_detected_image_data[i * width + j] = static_cast<unsigned char>(image(i, j));
                noisyMatrix[i * width + j] = static_cast<unsigned char>(noisy_image(i, j));
            }
        }
    
        // Save the images using stb_image_write
        stbi_write_png("smoothed_image.png", width, height, 1, smoothed_image_data, width);
        stbi_write_png("sharpened_image.png", width, height, 1, sharpened_image_data, width);
        stbi_write_png("edge_detected_image.png", width, height, 1, edge_detected_image_data, width);
        stbi_write_png("noisy_image.png", width, height, 1, noisyMatrix, width);
    
        // Free allocated memory
        delete[] smoothed_image_data;
        delete[] sharpened_image_data;
        delete[] edge_detected_image_data;
        delete[] noisyMatrix;
    
        return 0;
    }
    // Compute the Kronecker product to form the matrix A_1
    Eigen::SparseMatrix<double> A1 = buildSparseConvolutionMatrix(smoothing, height, width);
    
    // Flatten the image matrix to a vector
   // Eigen::VectorXd v = Eigen::Map<Eigen::VectorXd>(image.data(), image.size());
    
    // Now create a vector which is the product between the sparse matrix A1 and the vector v
    Eigen::VectorXd Av = A1 * v;
    
   // std::cout<<Av<<std::endl;
    
    Eigen::MatrixXd smoothed_image = Eigen::Map<Eigen::MatrixXd>(Av.data(), height, width);
     std::cout<<Av.norm()<<std::endl;


    // Convertire la matrice smoothed_image in un array di unsigned char per salvarla
    unsigned char* smoothed_image_data = new unsigned char[width * height];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
        double value = smoothed_image(i, j);  // Get the original double value from the matrix
        // Apply clamping directly within the loop
        if (value < 0.0) value = 0.0;        // Clamp the minimum value
        else if (value > 255.0) value = 255.0; // Clamp the maximum value
        
        // Convert the clamped double to unsigned char
        smoothed_image_data[i * width + j] = static_cast<unsigned char>(value);
        }   
    }

    // Salvare l'immagine smoothed usando stb_image_write
    stbi_write_png("smoothed_image.png", width, height, 1, smoothed_image_data, width);

    // Liberare la memoria allocata
    delete[] smoothed_image_data;
    // Convert the result back to a 2D matrix
      

  /*  for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            smoothed_image_data[i * width + j] = static_cast<unsigned char>(smoothed_image(i, j));
            sharpened_image_data[i * width + j] = static_cast<unsigned char>(sharpened_image(i, j));
            edge_detected_image_data[i * width + j] = static_cast<unsigned char>(edge_detected_image(i, j));
            noisyMatrix[i * width + j] = static_cast<unsigned char>(noisy_image(i, j));
        }
    }

    // Save the images using stb_image_write
    stbi_write_png("smoothed_image.png", width, height, 1, smoothed_image_data, width);
    stbi_write_png("sharpened_image.png", width, height, 1, sharpened_image_data, width);
    stbi_write_png("edge_detected_image.png", width, height, 1, edge_detected_image_data, width);
    stbi_write_png("noisy_image.png", width, height, 1, noisyMatrix, width);

    // Free allocated memory
    delete[] smoothed_image_data;
    delete[] sharpened_image_data;
    delete[] edge_detected_image_data;
    delete[] noisyMatrix;*/

    
}




