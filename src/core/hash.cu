#include <voxhash/core/block.h>
#include <voxhash/core/hash.h>

namespace voxhash {

// template <typename BlockType>
// bool CPUHashStrategy<BlockType>::findValue(
//         const Index3D key, typename BlockType::Ptr& value) const {
//     const auto it = hash_.find(key);
//     if (it == hash_.end()) return false;
//     value = it->second;
//     return true;
// }

// template <typename BlockType>
// void CPUHashStrategy<BlockType>::findValues(
//         const std::vector<Index3D> keys,
//         std::vector<typename BlockType::Ptr>& values,
//         std::vector<bool>& found) const {
//     found.resize(keys.size());
//     values.resize(keys.size());

//     for (size_t idx = 0; idx < keys.size(); idx++) found[idx] = findValue(keys[idx],
//     values[idx]);
// }

// template <typename BlockType>
// std::vector<Index3D> CPUHashStrategy<BlockType>::getAllKeys() const {
//     std::vector<Index3D> indices;
//     indices.reserve(hash_.size());

//     for (const auto& kv : hash_) {
//         indices.push_back(kv.first);
//     }
//     return indices;
// }

// template <typename BlockType>
// std::vector<typename BlockType::Ptr> CPUHashStrategy<BlockType>::getAllValues() const {
//     std::vector<typename BlockType::Ptr> ptrs;
//     ptrs.reserve(hash_.size());

//     for (const auto& kv : hash_) {
//         ptrs.push_back(kv.second);
//     }
//     return ptrs;
// }

// template <typename BlockType>
// bool CPUHashStrategy<BlockType>::insertValue(
//         const Index3D key, const typename BlockType::Ptr value) {
//     const auto it = hash_.find(key);
//     if (it != hash_.end())  // Reject insertion if a value already exists
//         return value == it->second;
//     hash_[key] = value;
//     return true;
// }

// template <typename BlockType>
// void CPUHashStrategy<BlockType>::insertValues(
//         const std::vector<Index3D> keys,
//         const std::vector<typename BlockType::Ptr> values,
//         std::vector<bool>& inserted) {
//     inserted.resize(keys.size());

//     for (size_t idx = 0; idx < keys.size(); idx++)
//         inserted[idx] = insertValue(keys[idx], values[idx]);
// }

// template <typename BlockType>
// bool CPUHashStrategy<BlockType>::eraseValue(const Index3D key) {
//     const auto it = hash_.find(key);
//     if (it == hash_.end()) return true;
//     hash_.erase(it);
//     return true;
// }

// template <typename BlockType>
// void CPUHashStrategy<BlockType>::eraseValues(
//         const std::vector<Index3D> keys, std::vector<bool>& erased) {
//     erased.resize(keys.size());

//     for (size_t idx = 0; idx < keys.size(); idx++) erased[idx] = eraseValue(keys[idx]);
// }

template <typename BlockType>
GPUHashStrategy<BlockType>::GPUHashStrategy(
        std::shared_ptr<CudaStream> cuda_stream, size_t num_increase_objects, MemoryType type)
    : HashStrategy<BlockType>(cuda_stream),
      num_increase_objects_(num_increase_objects),
      current_objects_(num_increase_objects),
      type_(type) {
    hash_ = GPUHashMapType<BlockType>::createDeviceObject(current_objects_);
}

template <typename BlockType>
GPUHashStrategy<BlockType>::~GPUHashStrategy() {
    GPUHashMapType<BlockType>::destroyDeviceObject(hash_);
}

template <typename BlockType>
__global__ void findValuesKernel(
        Index3D* keys,
        typename BlockType::VoxelType** values,
        Bool* found,
        GPUHashMapType<BlockType> map,
        size_t num_keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_keys) return;
    Index3D key = keys[i];

    auto it = map.find(key);

    if (it != map.end()) {
        values[i] = it->second;
        found[i] = 1;
    } else {
        values[i] = nullptr;
        found[i] = 0;
    }
}

template <typename BlockType>
__global__ void insertValuesKernel(
        Index3D* keys,
        typename BlockType::VoxelType** values,
        Bool* inserted,
        GPUHashMapType<BlockType> map,
        size_t num_keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_keys) return;

    auto result = map.emplace(keys[i], values[i]);
    inserted[i] = result.second ? 1 : 0;
}

template <typename BlockType>
std::pair<Vector<Bool>, Vector<typename BlockType::VoxelType*>>
GPUHashStrategy<BlockType>::findValues(const Vector<Index3D>& keys) const {
    size_t num_queries = keys.size();

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = nullptr;
    if (keys.location() == type_) {
        keys_data = keys.data();
    } else {
        keys_ptr = Vector<Index3D>::copyFrom(keys, type_, *this->stream_);
        keys_data = keys_ptr->data();
    }

    Vector<typename BlockType::VoxelType*> values(num_queries, type_);
    Vector<Bool> found(num_queries, type_);

    int bpg = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    findValuesKernel<BlockType><<<bpg, THREADS_PER_BLOCK, 0, *this->stream_>>>(
            keys_data, values.data(), found.data(), hash_, num_queries);
    CUDA_CHECK(cudaGetLastError());
    this->stream_->synchronize();

    return std::make_pair(std::move(found), std::move(values));
}

template <typename BlockType>
Vector<Bool> GPUHashStrategy<BlockType>::insertValues(
        const Vector<Index3D>& keys, const Vector<typename BlockType::VoxelType*>& values) {
    size_t num_queries = keys.size();

    // Check Load factor and recreate hash_ if total size exceeds current_objects
    if (hash_.size() + num_queries > current_objects_)
        recreateHash(
                std::max(num_queries + current_objects_, current_objects_ + num_increase_objects_));

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = nullptr;
    if (keys.location() == type_) {
        keys_data = keys.data();
    } else {
        keys_ptr = Vector<Index3D>::copyFrom(keys, type_, *this->stream_);
        keys_data = keys_ptr->data();
    }
    std::shared_ptr<Vector<typename BlockType::VoxelType*>> values_ptr = nullptr;
    typename BlockType::VoxelType** values_data = nullptr;
    if (values.location() == type_) {
        values_data = values.data();
    } else {
        values_ptr =
                Vector<typename BlockType::VoxelType*>::copyFrom(values, type_, *this->stream_);
        values_data = values_ptr->data();
    }

    Vector<Bool> inserted(num_queries, type_);

    int bpg = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    insertValuesKernel<BlockType><<<bpg, THREADS_PER_BLOCK, 0, *this->stream_>>>(
            keys_data, values_data, inserted.data(), hash_, num_queries);
    CUDA_CHECK(cudaGetLastError());
    this->stream_->synchronize();

    return inserted;
}

template <typename BlockType>
struct extract_kv {
    __host__ __device__ thrust::tuple<Index3D, typename BlockType::VoxelType*> operator()(
            const stdgpu::pair<const Index3D, typename BlockType::VoxelType*>& p) const {
        return thrust::make_tuple(p.first, p.second);
    }
};

template <typename BlockType>
std::pair<Vector<Index3D>, Vector<typename BlockType::VoxelType*>>
GPUHashStrategy<BlockType>::getAllKeyValues() const {
    Vector<Index3D> keys(hash_.size(), type_);
    Vector<typename BlockType::VoxelType*> values(hash_.size(), type_);

    auto range_map = hash_.device_range();
    thrust::transform(
            range_map.begin(),
            range_map.end(),
            thrust::make_zip_iterator(thrust::make_tuple(keys.data(), values.data())),
            extract_kv<BlockType>());
    return std::make_pair(std::move(keys), std::move(values));
}

template <typename BlockType>
void GPUHashStrategy<BlockType>::recreateHash(size_t increase_factor) {
    std::cout << "Recreating hash with " << increase_factor << " objects\n";
    GPUHashMapType<BlockType> new_hash =
            GPUHashMapType<BlockType>::createDeviceObject(increase_factor);

    auto result = getAllKeyValues();
    size_t num_keys = result.first.size();
    if (num_keys > 0) {
        Vector<Bool> inserted(num_keys, type_);

        int bpg = (num_keys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        insertValuesKernel<BlockType><<<bpg, THREADS_PER_BLOCK, 0, *this->stream_>>>(
                result.first.data(), result.second.data(), inserted.data(), new_hash, num_keys);
        CUDA_CHECK(cudaGetLastError());
        this->stream_->synchronize();
    }

    GPUHashMapType<BlockType> old_hash = hash_;
    hash_ = new_hash;
    GPUHashMapType<BlockType>::destroyDeviceObject(old_hash);
    current_objects_ = increase_factor;
}

template <typename BlockType>
__global__ void eraseKeysKernel(
        Index3D* keys, Bool* erased, GPUHashMapType<BlockType> map, size_t num_keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_keys) return;
    erased[i] = map.erase(keys[i]);
}

template <typename BlockType>
Vector<Bool> GPUHashStrategy<BlockType>::eraseValues(const Vector<Index3D>& keys) {
    size_t num_queries = keys.size();

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = nullptr;
    if (keys.location() == type_) {
        keys_data = keys.data();
    } else {
        keys_ptr = Vector<Index3D>::copyFrom(keys, type_, *this->stream_);
        keys_data = keys_ptr->data();
    }

    Vector<Bool> erased(num_queries, type_);

    if (num_queries > 0) {
        int bpg = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        eraseKeysKernel<BlockType><<<bpg, THREADS_PER_BLOCK, 0, *this->stream_>>>(
                keys_data, erased.data(), hash_, num_queries);
        CUDA_CHECK(cudaGetLastError());
        this->stream_->synchronize();
    }
    return erased;
}

template class HashStrategy<Block<int>>;
template class HashStrategy<Block<float>>;

// template class CPUHashStrategy<Block<int>>;
// template class CPUHashStrategy<Block<float>>;

template class GPUHashStrategy<Block<int>>;
template class GPUHashStrategy<Block<float>>;
// TODO: Add Block Type Strategies
}  // namespace voxhash