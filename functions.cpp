//needed functions
// convolve relu  maxpool flattern fully_connected softmax
#include "teku.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>





// Function to load images from a CSV file
// Each line in the CSV file represents a flattened 28x28 image with pixel values separated by commas.
// The function reads the file, normalizes the pixel values to [0, 1], and converts them into a 2D matrix format.
// The function returns a vector of images i.e. set of images.
ImageSet load_images_from_csv(const string& filename) {
    ifstream file(filename);
    string line;
    ImageSet images;

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<float> pixels;

        // Read 784 values (flattened image)
        while (getline(ss, value, ',')) {
            pixels.push_back(stof(value) / 255.0f); // Normalize to [0, 1]
        }

        // Convert to 28x28
        Image image(28, vector<float>(28));
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                image[i][j] = pixels[i * 28 + j];
            }
        }

        images.push_back(image);
    }

    return images;
}

//CONVOLVE FUNCTION
Image convolve(const Image& image, const Image& filter) {
    int H = image.size();
    int W = image[0].size();
    int KH = filter.size();
    int KW = filter[0].size();

    int OH = H - KH + 1;
    int OW = W - KW + 1;

    Image output(OH, std::vector<float>(OW));

    for (int i = 0; i < OH; ++i) {
        for (int j = 0; j < OW; ++j) 
        {
            //aile samma eauta image ko eauta specific pixel choose hgarya xam i,j wala
            float sum = 0.0;   // tesko value chai 0 initialize garyo
            for (int ki = 0; ki < KH; ++ki) 
            {
                for (int kj = 0; kj < KW; ++kj) 
                {
                    sum += image[i + ki][j + kj] * filter[ki][kj];
                }
            }
            output[i][j] = sum;    // mathi initialize gareko sum ma output store garyo
        }
    }

    return output;
}





//RELU FUNCTION
Image relu(const Image& image) 
{
    Image output = image; // Create a copy of the input image
    // Apply ReLU activation function
    // ReLU: f(x) = max(0, x)
    // This replaces all negative values with 0 and keeps positive values unchanged.
    int H = image.size();
    int W = image[0].size();
    for(int i = 0; i < H; ++i)
    {
        for(int j = 0; j < W; ++j)
        {
            if(output[i][j] < 0) {
                output[i][j] = 0; // Set negative values to 0
            }
            // Positive values remain unchanged. No need to explicitly set them, as they are already positive
        }
    }
    return output; // Return the modified image with ReLU applied
    
}





//MAXPOOL FUNCTION
Image maxpool(const Image& image, int pool_size)  // pool size=2 for 2x2 pooling
{
    int H = image.size();
    int W = image[0].size();
    int OH = H / pool_size;
    int OW = W / pool_size;

    Image output(OH, std::vector<float>(OW));

    for (int i = 0; i < OH; ++i) {
        for (int j = 0; j < OW; ++j) {
            float max_val = -INFINITY;
            for (int pi = 0; pi < pool_size; ++pi) {
                for (int pj = 0; pj < pool_size; ++pj) {
                    int r = i * pool_size + pi;
                    int c = j * pool_size + pj;  
                //Compute the actual row (r) and column (c) in the original image corresponding to the current pooling window cell.
                    max_val = std::max(max_val, image[r][c]);
                }
            }
            output[i][j] = max_val;
        }
    }

    return output;
}



//FLATTEN FUNCTION
std::vector<float> flatten(const Image& image) 
{
    std::vector<float> flat; // Create a flat vector to hold the values
    // Iterate through each row and each value in the row to flatten the 2D image into a 1D vector
    int H = image.size();
    int W = image[0].size();
    for(int i = 0; i < H; ++i)
    {
        for(int j = 0; j < W; ++j)
        {
            flat.push_back(image[i][j]); // Add each value to the flat vector}
        }
    }
    return flat;

}


// FULLY CONNECTED LAYER FUNXCTON
// This function takes a flattened input vector, weights matrix, and biases vector to compute the output of a fully connected layer.
// It performs the dot product of the input with the weights and adds the biases.
std::vector<float> fully_connected(const std::vector<float>& input,
                                   const std::vector<std::vector<float>>& weights,
                                   const std::vector<float>& biases) 
{
    int output_size = weights.size();       // number of neurons
    int input_size = input.size();          // flattened input size

    std::vector<float> output(output_size, 0.0f);    // Initialize output values to 0

    for (int i = 0; i < output_size; ++i) 
    {
        for (int j = 0; j < input_size; ++j) 
        {
            output[i] += weights[i][j] * input[j];  // Dot product of input and weights for each neuron
            // This computes the weighted sum of inputs for each neuron in the fully connected layer.
        }
        output[i] += biases[i]; // add bias
    }

    return output;  // This returns the vector of raw output values (also called logits) from the fully connected layer.
    
}




// Softmax function to convert logits to probabilities
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> exp_values(logits.size());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        exp_values[i] = exp(logits[i]);   // Don't use push_back
        sum_exp += exp_values[i];
    }
    std::vector<float> probabilities(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = exp_values[i] / sum_exp;
    }
return probabilities;
// This function takes a vector of logits (raw output values from the fully connected layer) and converts them into probabilities using the softmax function.
// It first computes the exponentials of the logits, sums them up, and then normalizes each exponential by dividing it by the total sum to get probabilities.




// Forward pass function that processes a set of images through the CNN
// It applies convolution, ReLU, max pooling, flattening, and fully connected layers to compute class probabilities for each image.

// bahira bata pathauda bulk ma pathaune tara vitra ko each function ma pathauda chai one by one pathaune loop layera.

std::vector<std::vector<float>> forward_pass(
    const ImageSet& images,
    const ImageSet& filters,
    const std::vector<std::vector<float>>& fc_weights,
    const std::vector<float>& fc_biases
) {
    std::vector<std::vector<float>> results; // Stores softmax outputs for all images i.e. probabilities for each class

    for (const auto& image : images) {    // Iterate through each image in the dataset
        


        // 1. Apply all filters and collect conv outputs

        std::vector<Image> conv_outputs;  // variable to Store outputs for an image from convolution layer for all filter
        // Loop through each filter and apply convolution
        for (const auto& filter : filters) {
            conv_outputs.push_back(convolve(image, filter));
        }

        // 2. ReLU for each filter output
        for (auto& conv : conv_outputs) {   // eauta filter ko output lai ekchoti relu vitra pathaune ako result lai naya variable ma narakhi puranai conv_outputs vanne variable nai update garne
            conv = relu(conv);
        }




        // aile samma bahira ko loop (loop each image) bata choose vako image lai sabai filter ma pathayera ako feature maps lai relu ma pathayera ako output lai conv_outputs(set of feature maps for single image) ma store gareko xa.





        // 3. MaxPool (2x2) for each feature map
        std::vector<Image> pooled_outputs;     // variable to store pooled outputs for each feature maps of that specific image

        for (const auto& relu_out : conv_outputs) {  //one by one sabai relu outputs lai maxpool ma pathaune
            pooled_outputs.push_back(maxpool(relu_out, 2));
        }

        // This applies max pooling to each feature map, reducing its size by half (2x2 pooling).
        // pooled_outputs now contains the max-pooled outputs for each feature map of the current image.
        // Each pooled output is a smaller image (e.g., 13x13 if the original was 26x26).



        // 4. Flatten all pooled outputs into a single 1D vector

        //pooled_outputs contains multiple 2D matrices → one for each filter applied to that single image.
        /*   for single image we have multiple pooled outputs i.e. equal to no of filters we used previously 
        so  this function flatten all the pooled outputs of that image and then join them end to end to get single 1D vector */


        std::vector<float> flattened_input;      // 1D variable to store flattened input for fully connected layer
        for (const auto& pooled : pooled_outputs) {  // Loop through each pooled output and flatten it

            std::vector<float> flat = flatten(pooled);
            flattened_input.insert(flattened_input.end(), flat.begin(), flat.end());   // Append all the flattened outputs to make single 1D vector
        } 
        // flattened_input now contains all the pixel values from all pooled outputs concatenated into a single vector.
        // This vector will be used as input to the fully connected layer.  ( size of that flattened input for single input will be  13 * 13 * no_of_filters )

        // 5. Fully Connected Layer
        std::vector<float> fc_output = fully_connected(flattened_input, fc_weights, fc_biases);

        // 6. Softmax
        std::vector<float> probabilities = softmax(fc_output);

        // Store result
        results.push_back(probabilities);
    }

    return results; // each row = class probabilities for each image
}




// FOR BACKPROPAGATION OF FC LAYER
// This function computes the gradient of the loss with respect to the logits (raw output values from the fully connected layer).
// It takes the predicted probabilities and the true label as inputs and returns the gradient vector.
std::vector<float> softmax_grad(const std::vector<float>& probs, int label) {
    std::vector<float> grad = probs;
    grad[label] -= 1.0f;  // dL/dz = p - y
    return grad;
}

// Backward pass for the fully connected layer
// This function updates the weights and biases of the fully connected layer based on the gradients computed during backpropagation.
// It takes the input vector, gradient of the output, weights, biases, and learning rate as inputs.
void fc_backward(const std::vector<float>& input,
                 const std::vector<float>& grad_output,  // dL/dz
                 std::vector<std::vector<float>>& weights,
                 std::vector<float>& biases,
                 float learning_rate)
{
    int output_size = weights.size();
    int input_size = input.size();

    // Gradient for weights and biases
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] -= learning_rate * grad_output[i] * input[j];  // dW = dL/dz * input
        }
        biases[i] -= learning_rate * grad_output[i];  // dB = dL/dz
    }
}




