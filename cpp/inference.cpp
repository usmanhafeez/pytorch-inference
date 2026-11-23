
#include <torch/script.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main() {
    try {
        at::set_num_threads(1);

        auto model = torch::jit::load("models/fashion_cnn.pt");
        model.eval();

        // Read 28x28 float32 image from Python
        std::vector<float> pixels(28 * 28);

        std::ifstream file(
            "models/test_input.bin",
            std::ios::binary
        );

        if (!file) {
            std::cerr << "Could not open input file\n";
            return 1;
        }

        file.read(
            reinterpret_cast<char*>(pixels.data()),
            static_cast<std::streamsize>(
                pixels.size() * sizeof(float)
            )
        );

        if (!file) {
            std::cerr << "Could not read complete input image\n";
            return 1;
        }

        // Copy pixel data into an owned PyTorch tensor
        auto input = torch::from_blob(
            pixels.data(),
            {1, 1, 28, 28},
            torch::kFloat32
        ).clone();

        torch::InferenceMode guard;

        std::vector<torch::jit::IValue> inputs{input};

        auto output = model.forward(inputs).toTensor();

        auto probabilities = torch::softmax(output, 1);

        int predicted_class = output.argmax(1).item<int>();

        const std::vector<std::string> class_names = {
            "T-shirt/top", "Trouser", "Pullover", "Dress",
            "Coat", "Sandal", "Shirt", "Sneaker",
            "Bag", "Ankle boot"
        };

        float confidence =
            probabilities[0][predicted_class].item<float>() * 100;

        std::cout << "=== C++ Fashion-MNIST Inference ===\n";
        std::cout << "Input shape: " << input.sizes() << "\n";
        std::cout << "Output shape: " << output.sizes() << "\n";
        std::cout << "Predicted class: "
                  << class_names.at(predicted_class) << "\n";
        std::cout << "Confidence: " << confidence << "%\n";
        std::cout << "C++ inference successful!\n";

    } catch (const c10::Error& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}

