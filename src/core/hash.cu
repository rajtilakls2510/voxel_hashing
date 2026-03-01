#include <voxhash/core/block.h>
#include <voxhash/core/hash.h>

namespace voxhash {

template <typename BlockType>
bool CPUHashStrategy<BlockType>::findValue(
        const Index3D key, typename BlockType::Ptr& value) const {
    const auto it = hash_.find(key);
    if (it == hash_.end()) return false;
    value = it->second;
    return true;
}

template <typename BlockType>
void CPUHashStrategy<BlockType>::findValues(
        const std::vector<Index3D> keys,
        std::vector<typename BlockType::Ptr>& values,
        std::vector<bool>& found) const {
    found.resize(keys.size());
    values.resize(keys.size());

    for (size_t idx = 0; idx < keys.size(); idx++) found[idx] = findValue(keys[idx], values[idx]);
}

template <typename BlockType>
std::vector<Index3D> CPUHashStrategy<BlockType>::getAllKeys() const {
    std::vector<Index3D> indices;
    indices.reserve(hash_.size());

    for (const auto& kv : hash_) {
        indices.push_back(kv.first);
    }
    return indices;
}

template <typename BlockType>
std::vector<typename BlockType::Ptr> CPUHashStrategy<BlockType>::getAllValues() const {
    std::vector<typename BlockType::Ptr> ptrs;
    ptrs.reserve(hash_.size());

    for (const auto& kv : hash_) {
        ptrs.push_back(kv.second);
    }
    return ptrs;
}

template <typename BlockType>
bool CPUHashStrategy<BlockType>::insertValue(
        const Index3D key, const typename BlockType::Ptr value) {
    const auto it = hash_.find(key);
    if (it != hash_.end())  // Reject insertion if a value already exists
        return value == it->second;
    hash_[key] = value;
    return true;
}

template <typename BlockType>
void CPUHashStrategy<BlockType>::insertValues(
        const std::vector<Index3D> keys,
        const std::vector<typename BlockType::Ptr> values,
        std::vector<bool>& inserted) {
    inserted.resize(keys.size());

    for (size_t idx = 0; idx < keys.size(); idx++)
        inserted[idx] = insertValue(keys[idx], values[idx]);
}

template <typename BlockType>
bool CPUHashStrategy<BlockType>::eraseValue(const Index3D key) {
    const auto it = hash_.find(key);
    if (it == hash_.end()) return true;
    hash_.erase(it);
    return true;
}

template <typename BlockType>
void CPUHashStrategy<BlockType>::eraseValues(
        const std::vector<Index3D> keys, std::vector<bool>& erased) {
    erased.resize(keys.size());

    for (size_t idx = 0; idx < keys.size(); idx++) erased[idx] = eraseValue(keys[idx]);
}

template <typename BlockType>
GPUHashStrategy<BlockType>::GPUHashStrategy(
        std::shared_ptr<CudaStream> cuda_stream, size_t num_increase_objects, MemoryType type)
    : HashStrategy<BlockType>(cuda_stream),
      num_increase_objects_(num_increase_objects),
      current_objects_(num_increase_objects),
      type_(type) {
    hash_ = GPUIndex3DHashMapType<BlockType>::createDeviceObject(current_objects_);
}

template <typename BlockType>
GPUHashStrategy<BlockType>::~GPUHashStrategy() {
    GPUIndex3DHashMapType<BlockType>::destroyDeviceObject(hash_);
}

template <typename BlockType>
__global__ void findSingleValue(
        const Index3D key,
        typename BlockType::VoxelType* out_value,
        int* out_found,
        GPUIndex3DHashMapType<BlockType> map) {
    auto it = map.find(key);

    if (it != map.end()) {
        *out_value = it->second;
        *out_found = 1;
    } else {
        *out_found = 0;
    }
}

template <typename BlockType>
bool GPUHashStrategy<BlockType>::findValue(
        const Index3D key, typename BlockType::Ptr& value) const {
    typename BlockType::VoxelType* d_value = nullptr;
    int* d_found = nullptr;
    if (type_ == MemoryType::kUnified) {
        CUDA_CHECK(cudaMallocManaged(
                &d_value, sizeof(typename BlockType::VoxelType), cudaMemAttachGlobal));
        CUDA_CHECK(cudaMallocManaged(&d_found, sizeof(int), cudaMemAttachGlobal));
    } else {
        CUDA_CHECK(
                cudaMallocAsync(&d_value, sizeof(typename BlockType::VoxelType), *this->stream_));
        CUDA_CHECK(cudaMallocAsync(&d_found, sizeof(int), *this->stream_));
    }
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(int)));

    findSingleValue<BlockType><<<1, 1>>>(key, d_value, d_found, hash_);
    CUDA_CHECK(cudaGetLastError());
    this->stream_->synchronize();

    int h_found;
    CUDA_CHECK(cudaMemcpyAsync(&h_found, d_found, sizeof(int), cudaMemcpyDefault, *this->stream_));
    return h_found == 1;
}

template class HashStrategy<Block<int>>;
template class HashStrategy<Block<float>>;

template class CPUHashStrategy<Block<int>>;
template class CPUHashStrategy<Block<float>>;

template class GPUHashStrategy<Block<int>>;
template class GPUHashStrategy<Block<float>>;
// TODO: Add Block Type Strategies
}  // namespace voxhash