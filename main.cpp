#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm> // for std::shuffle
#include <random>    // for std::default_random_engine
#include <chrono>    // for seeding with time
#include "functions.h"

#include <cstdlib> // for rand(), RAND_MAX
#include <ctime>   // for seeding rand()
using namespace std;

const int EPOCHS= 5; // Number of epochs for training
// Your typedefs or using aliases if any
using Image = vector<vector<float>>;
using ImageSet = vector<Image>;

float random_float() {
    return ((float)rand() / RAND_MAX) * 0.2f - 0.1f;  // [-0.1, 0.1];  // Random float between -1 and 1
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


// Load the dataset (assuming your data is in "mnist_train.csv")
    LabeledDataset dataset = load_labeled_images_from_csv("mnist_train.csv");
    
    std::cout << "Total images loaded: " << dataset.images.size() << "\n";
    std::cout << "Total labels loaded: " << dataset.labels.size() << "\n";
    
    // You can now use dataset.images and dataset.labels for your CNN

// now to create a combined dataset of images and their corresponding labels
// we will create a vector of ImageSet and a vector of labels
// where each label corresponds to the category of the image (0 for cat, 1 for dog,.... etc.)




/*  
ImageSet all_images;
vector<int> labels;


for (auto& img : cat_images) {
    all_images.push_back(img);
    labels.push_back(0); // label 0 for cat
}

for (auto& img : dog_images) {
    all_images.push_back(img);
    labels.push_back(1); // label 1 for dog
} */




int num_filters = 2; // Number of filters for the convolution layer

    // Fix filters declaration
std::vector<Image> filters(num_filters); // Declare filters as a vector of Image
for (int i = 0; i < num_filters; ++i) 
{
    filters[i] = random_filter();
}

// Initialize random seed
srand(static_cast<unsigned int>(time(0)));   //C++'s random number generator (rand()) always produces the same sequence of numbers unless you seed it with something different

// Define the fully connected layer weights and biases
int num_classes = 10; // Number of output classes (e.g., cat and dog)
// pooled height and pooled width sould be 13 and 13.  csuse while passing 28*28 into convolution it become 26*26 the after passing into maxpol it become half
int pooled_height = 13; // After max pooling
int pooled_width = 13; // After max pooling
// Flattened size will be 13 * 13 * num_filters



int flattened_size = num_filters * pooled_height * pooled_width;
std::vector<std::vector<float>> fc_weights = random_weights(num_classes, flattened_size);
std::vector<float> fc_biases = random_biases(num_classes);



// Now we can perform the forward pass for each image in the dataset
// This will convolve each image with the filters, apply ReLU, flatten the output, and then pass it through the fully connected layer.
std::cout << "Starting forward pass for all images..." << std::endl;
// Perform forward pass for each image in the dataset

int correct = 0; // Counter for correct predictions
float total_loss = 0.0f; // Variable to accumulate loss
float learning_rate = 0.01; // Learning rate for the fully connected layer
int total = dataset.images.size();


// Removed local ForwardResult struct definition to use the one from functions.h





// Loop through each epoch
for(int i=0; i<EPOCHS; i++)
{
    /* 
    // Create combined vector of (image, label)
    std::vector<std::pair<Image, int>> combined;
    for (size_t i = 0; i < all_images.size(); ++i) {
        combined.push_back({all_images[i], labels[i]});
    }

    // Shuffle
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(combined.begin(), combined.end(), std::default_random_engine(seed));

    // Reassign to original vectors
    all_images.clear();
    labels.clear();
    for (auto& pair : combined) {
        all_images.push_back(pair.first);
        labels.push_back(pair.second);
    }

    // Now all_images and labels are shuffled together

*/


    std::cout << "Epoch " << i + 1 << ": Forward pass for all images..." << std::endl;

    for (int j = 0; j < dataset.images.size(); ++j)
    {

        ForwardResult result = forward_pass(dataset.images[j], filters, fc_weights, fc_biases);

        // ======= Loss + Accuracy =======
        float loss = cross_entropy(result.probabilities, dataset.labels[j]); 
        total_loss += loss;

        int pred = argmax(result.probabilities);
        if (pred == dataset.labels[j])
        {
            correct++;
        }

        // ======= Backward Pass (FC only) =======
        backward_pass_fc(result.flattened_input, result.probabilities, dataset.labels[j], fc_weights, fc_biases, learning_rate);

    }
    float accuracy = (float)correct / total * 100.0f;
    std::cout << "Epoch " << i+ 1 << " - Accuracy: " << accuracy << "%\n"; // Print accuracy for the epoch
    std::cout << "Epoch " << i + 1 << " - Average Loss: " << total_loss / total << "\n"; // average loss for the epoch
    correct = 0; // Reset correct count for the next epoch
    total_loss = 0.0f; // Reset total loss for the next epoch  
}
}
