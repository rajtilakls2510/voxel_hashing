
#include <voxhash/core/cuda_utils.h>

#include <iostream>

using namespace voxhash;

int main(int argc, char* argv[]) {
    std::cout << "Warming up Cuda\n";

    try {
        warmupCuda();
        std::cout << "Cuda warmed up\n";
    } catch (const CudaError& e) {
        std::cerr << "Failed to warmup cuda: " << e.message().c_str() << "\n";
    }
}