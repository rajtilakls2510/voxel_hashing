
#include <iostream>

#include "voxhash/core/cuda_utils.h"

namespace voxhash {
void check_cuda_error_value(
        cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
                  << line << " '" << func << "'. Error string: " << cudaGetErrorString(result)
                  << ".\n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

void warmupCuda() {
    warm_up_gpu<<<64, 128>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaStreamAsync::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(*stream_ptr_)); }

CudaStreamOwning::~CudaStreamOwning() {
    try {
        this->synchronize();
    } catch (const CudaError& e) {
    }
    try {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    } catch (const CudaError& e) {
    }
}

CudaStreamOwning::CudaStreamOwning(const unsigned int flags) : CudaStreamAsync(&stream_) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

void DefaultStream::synchronize() const { CUDA_CHECK(cudaStreamSynchronize(default_stream_)); }

std::shared_ptr<CudaStream> CudaStream::createCudaStream(CudaStreamType stream_type) {
    switch (stream_type) {
        case CudaStreamType::kLegacyDefault:
            return std::make_shared<DefaultStream>(cudaStreamLegacy);
        case CudaStreamType::kBlocking:
            return std::make_shared<CudaStreamOwning>(cudaStreamDefault);
        case CudaStreamType::kNonBlocking:
            return std::make_shared<CudaStreamOwning>(cudaStreamNonBlocking);
        case CudaStreamType::kPerThreadDefault:
            return std::make_shared<DefaultStream>(cudaStreamPerThread);
        default:
            throw std::invalid_argument("received unspported CudaStreamType!");
    }
}
}  // namespace voxhash