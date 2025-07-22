#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "teku.h"




#include <cstdlib> // for rand(), RAND_MAX
#include <ctime>   // for seeding rand()

float random_float() {
    return static_cast<float>(rand()) / RAND_MAX * 2 - 1;  // Random float between -1 and 1
}

Image random_filter() {
    Image filter(3, std::vector<float>(3));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            filter[i][j] = random_float();
    return filter;
}

std::vector<std::vector<float>> random_weights(int output_size, int input_size) {
    std::vector<std::vector<float>> weights(output_size, std::vector<float>(input_size));
    for (auto& row : weights)
        for (auto& val : row)
            val = random_float();
    return weights;
}

std::vector<float> random_biases(int size) {
    std::vector<float> biases(size);
    for (auto& b : biases)
        b = random_float();
    return biases;
}



int main()
{
// call the function each time to load images from the CSV files for different categories and create a seperate image set for them.
ImageSet cat_images = load_images_from_csv("cat.csv");
ImageSet dog_images = load_images_from_csv("dog.csv");

// now to create a combined dataset of images and their corresponding labels
// we will create a vector of ImageSet and a vector of labels
// where each label corresponds to the category of the image (0 for cat, 1 for dog,.... etc.)
ImageSet all_images;
vector<int> labels;

for (auto& img : cat_images) {
    all_images.push_back(img);
    labels.push_back(0); // label 0 for cat
}

for (auto& img : dog_images) {
    all_images.push_back(img);
    labels.push_back(1); // label 1 for dog
}
ImageSet filters[2]; // Assuming we have 2 filters for the convolution layer
// Each filter is a 3x3 matrix, and we can define them as follows:
// Horizontal edge detection filter



filters[0] = random_filter(); // Random filter for horizontal edges
filters[1] = random_filter(); // Random filter for vertical edges
// Initialize random seed
    srand(static_cast<unsigned int>(time(0)));  
// Define the fully connected layer weights and biases
int num_classes = 2; // Number of output classes (e.g., cat and dog)
std::vector<std::vector<float>> fc_weights = random_weights(num_classes, 784);
std::vector<float> fc_biases = random_biases(num_classes);



// Now we can perform the forward pass for each image in the dataset
// This will convolve each image with the filters, apply ReLU, flatten the output, and then pass it through the fully connected layer.
std::cout << "Starting forward pass for all images..." << std::endl;
// Perform forward pass for each image in the dataset






}
