#ifndef TEKU_H
#define TEKU_H

#include <vector>

using Image = vector<vector<float>>;           // the type of variable like int represents a single integer similarl it represents a 2D matrix i.e. 28x28 image
using ImageSet = vector<Image>;                // Set of images

Image convolve(const Image& image, const Image& filter);
Image relu(const Image& image);
Image maxpool(const Image& image, int pool_size);
std::vector<float> flatten(const Image& image);



std::vector<float> fully_connected(const std::vector<float>& input,
                                   const std::vector<std::vector<float>>& weights,
                                   const std::vector<float>& biases);



std::vector<std::vector<float>> forward_pass(
    const ImageSet& images,
    const ImageSet& filters,
    const std::vector<std::vector<float>>& fc_weights,
    const std::vector<float>& fc_biases
)

std::vector<float> softmax(const std::vector<float>& logits);

std::vector<float> softmax_grad(const std::vector<float>& probs, int label);

ImageSet load_images_from_csv(const std::string& filename);

void fc_backward(const std::vector<float>& input,
                 const std::vector<float>& grad_output,  // dL/dz
                 std::vector<std::vector<float>>& weights,
                 std::vector<float>& biases,
                 float learning_rate);

#endif
