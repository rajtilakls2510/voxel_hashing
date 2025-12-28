
#include <iostream>
#include <string>

#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/vector.h"

using namespace voxhash;

int main(int argc, char* argv[]) {
    warmupCuda();

    size_t size = 100;
    MemoryType type = MemoryType::kDevice;
    Vector<float> v(size, type);
    float test_data = 100;
    cudaMemcpy(v.data() + 1, &test_data, sizeof(float), cudaMemcpyDefault);

    Vector<float>::Ptr v1 = Vector<float>::copyFrom(v, MemoryType::kHost);

    std::cout << "Data (" << v1->size() << "): ";
    for (size_t i = 0; i < v1->size(); i++) {
        std::cout << v1->data()[i] << ",";
    }
    std::cout << "\n";

    std::cout << " Data at 1 before clearing: " << v1->data()[1] << "\n";
    v1->clear();
    std::cout << " Data at 1 after clearing: " << v1->data()[1] << "\n";

    bool isSet = v1->setFrom(v);
    if (!isSet)
        std::cout << "Not Set\n";
    else {
        std::cout << "Data (" << v1->size() << "): ";
        for (size_t i = 0; i < v1->size(); i++) {
            std::cout << v1->data()[i] << ",";
        }
        std::cout << "\n";
    }

    float* ptr = v.release();
    std::cout << "Valid: " << std::boolalpha << v.valid() << "\n";
    cudaFree(ptr);

    std::cout << "Created vector of size: " << size << " on " << to_string(type) << "\n";

    std::vector<int> vector_to_convert;
    for (int i = 0; i < 10; i++) vector_to_convert.push_back(i);
    Vector<int>::Ptr vector_converted_device =
            Vector<int>::copyFrom(vector_to_convert, MemoryType::kDevice);
    Vector<int>::Ptr vector_converted =
            Vector<int>::copyFrom(*vector_converted_device, MemoryType::kHost);

    std::cout << "Vector: ";
    for (int i = 0; i < 10; i++) std::cout << vector_converted->data()[i] << ", ";
    std::cout << "\n";
}