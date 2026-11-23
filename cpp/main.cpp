
#include <iostream>
#include <torch/torch.h>

int main() {
    torch::Tensor a = torch::tensor({1.0, 2.0, 3.0});
    torch::Tensor b = torch::tensor({4.0, 5.0, 6.0});

    torch::Tensor result = a + b;

    std::cout << "Tensor A: " << a << std::endl;
    std::cout << "Tensor B: " << b << std::endl;
    std::cout << "A + B: " << result << std::endl;

    std::cout << "LibTorch C++ test successful!" << std::endl;

    return 0;
}

