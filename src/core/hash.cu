#include <voxhash/core/block.h>
#include <voxhash/core/hash.h>

namespace voxhash {

template <typename BlockType>
std::pair<Vector<Bool>, Vector<typename BlockType::VoxelType*>>
CPUHashStrategy<BlockType>::findValues(
        const Vector<Index3D>& keys, const CudaStream& stream) const {
    size_t num_queries = keys.size();

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = optionallyCopyDataPtr<Index3D>(keys, type_, keys_ptr, stream);

    Vector<typename BlockType::VoxelType*> values(num_queries, type_);
    Vector<Bool> found(num_queries, type_);

    for (size_t idx = 0; idx < num_queries; idx++) {
        const auto it = hash_.find(keys_data[idx]);
        if (it == hash_.end()) {
            found.data()[idx] = 0;
            values.data()[idx] = nullptr;
        } else {
            found.data()[idx] = 1;
            values.data()[idx] = it->second;
        }
    }

    return std::make_pair(std::move(found), std::move(values));
}

template <typename BlockType>
Vector<Bool> CPUHashStrategy<BlockType>::insertValues(
        const Vector<Index3D>& keys,
        const Vector<typename BlockType::VoxelType*>& values,
        const CudaStream& stream) {
    size_t num_queries = keys.size();

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = optionallyCopyDataPtr<Index3D>(keys, type_, keys_ptr, stream);
    std::shared_ptr<Vector<typename BlockType::VoxelType*>> values_ptr = nullptr;
    typename BlockType::VoxelType** values_data =
            optionallyCopyDataPtr<typename BlockType::VoxelType*>(
                    values, type_, values_ptr, stream);

    Vector<Bool> inserted(num_queries, type_);

    for (size_t idx = 0; idx < num_queries; idx++) {
        auto result = hash_.emplace(keys_data[idx], values_data[idx]);
        inserted[idx] = result.second ? 1 : 0;
    }

    return inserted;
}

template <typename BlockType>
std::pair<Vector<Index3D>, Vector<typename BlockType::VoxelType*>>
CPUHashStrategy<BlockType>::getAllKeyValues(const CudaStream& stream) const {
    Vector<Index3D> keys(hash_.size(), type_);
    Vector<typename BlockType::VoxelType*> values(hash_.size(), type_);

    size_t idx = 0;
    for (const auto& kv : hash_) {
        keys.data()[idx] = kv.first;
        values.data()[idx] = kv.second;
        ++idx;
    }

    return std::make_pair(std::move(keys), std::move(values));
}

template <typename BlockType>
Vector<Bool> CPUHashStrategy<BlockType>::eraseValues(
        const Vector<Index3D>& keys, const CudaStream& stream) {
    size_t num_queries = keys.size();

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = optionallyCopyDataPtr<Index3D>(keys, type_, keys_ptr, stream);

    Vector<Bool> erased(num_queries, type_);

    for (size_t idx = 0; idx < num_queries; idx++) {
        erased[idx] = hash_.erase(keys_data[idx]) ? 1 : 0;
    }

    return erased;
}

template <typename BlockType>
GPUHashStrategy<BlockType>::GPUHashStrategy(size_t num_increase_objects, MemoryType type)
    : num_increase_objects_(num_increase_objects),
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
struct extract_kv {
    __host__ __device__ thrust::tuple<Index3D, typename BlockType::VoxelType*> operator()(
            const stdgpu::pair<const Index3D, typename BlockType::VoxelType*>& p) const {
        return thrust::make_tuple(p.first, p.second);
    }
};

template <typename BlockType>
__global__ void eraseKeysKernel(
        Index3D* keys, Bool* erased, GPUHashMapType<BlockType> map, size_t num_keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_keys) return;
    erased[i] = map.erase(keys[i]);
}

template <typename BlockType>
std::pair<Vector<Bool>, Vector<typename BlockType::VoxelType*>>
GPUHashStrategy<BlockType>::findValues(
        const Vector<Index3D>& keys, const CudaStream& stream) const {
    size_t num_queries = keys.size();

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = optionallyCopyDataPtr<Index3D>(keys, type_, keys_ptr, stream);

    Vector<typename BlockType::VoxelType*> values(num_queries, type_);
    Vector<Bool> found(num_queries, type_);

    int bpg = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    findValuesKernel<BlockType><<<bpg, THREADS_PER_BLOCK, 0, stream>>>(
            keys_data, values.data(), found.data(), hash_, num_queries);
    CUDA_CHECK(cudaGetLastError());
    stream.synchronize();

    return std::make_pair(std::move(found), std::move(values));
}

template <typename BlockType>
Vector<Bool> GPUHashStrategy<BlockType>::insertValues(
        const Vector<Index3D>& keys,
        const Vector<typename BlockType::VoxelType*>& values,
        const CudaStream& stream) {
    size_t num_queries = keys.size();

    // Check Load factor and recreate hash_ if total size exceeds current_objects
    if (hash_.size() + num_queries > current_objects_)
        recreateHash(
                std::max(num_queries + current_objects_, current_objects_ + num_increase_objects_));

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = optionallyCopyDataPtr<Index3D>(keys, type_, keys_ptr, stream);
    std::shared_ptr<Vector<typename BlockType::VoxelType*>> values_ptr = nullptr;
    typename BlockType::VoxelType** values_data =
            optionallyCopyDataPtr<typename BlockType::VoxelType*>(
                    values, type_, values_ptr, stream);

    Vector<Bool> inserted(num_queries, type_);

    int bpg = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    insertValuesKernel<BlockType><<<bpg, THREADS_PER_BLOCK, 0, stream>>>(
            keys_data, values_data, inserted.data(), hash_, num_queries);
    CUDA_CHECK(cudaGetLastError());
    stream.synchronize();

    return inserted;
}

template <typename BlockType>
std::pair<Vector<Index3D>, Vector<typename BlockType::VoxelType*>>
GPUHashStrategy<BlockType>::getAllKeyValues(const CudaStream& stream) const {
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
void GPUHashStrategy<BlockType>::recreateHash(size_t increase_factor, const CudaStream& stream) {
    std::cout << "Recreating hash with " << increase_factor << " objects\n";
    GPUHashMapType<BlockType> new_hash =
            GPUHashMapType<BlockType>::createDeviceObject(increase_factor);

    auto result = getAllKeyValues(stream);
    size_t num_keys = result.first.size();
    if (num_keys > 0) {
        Vector<Bool> inserted(num_keys, type_);

        int bpg = (num_keys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        insertValuesKernel<BlockType><<<bpg, THREADS_PER_BLOCK, 0, stream>>>(
                result.first.data(), result.second.data(), inserted.data(), new_hash, num_keys);
        CUDA_CHECK(cudaGetLastError());
        stream.synchronize();
    }

    GPUHashMapType<BlockType> old_hash = hash_;
    hash_ = new_hash;
    GPUHashMapType<BlockType>::destroyDeviceObject(old_hash);
    current_objects_ = increase_factor;
}

template <typename BlockType>
Vector<Bool> GPUHashStrategy<BlockType>::eraseValues(
        const Vector<Index3D>& keys, const CudaStream& stream) {
    size_t num_queries = keys.size();

    std::shared_ptr<Vector<Index3D>> keys_ptr = nullptr;
    Index3D* keys_data = optionallyCopyDataPtr<Index3D>(keys, type_, keys_ptr, stream);

    Vector<Bool> erased(num_queries, type_);

    if (num_queries > 0) {
        int bpg = (num_queries + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        eraseKeysKernel<BlockType><<<bpg, THREADS_PER_BLOCK, 0, stream>>>(
                keys_data, erased.data(), hash_, num_queries);
        CUDA_CHECK(cudaGetLastError());
        stream.synchronize();
    }
    return erased;
}

template <typename T>
T* optionallyCopyDataPtr(
        const Vector<T>& vec,
        MemoryType target_type,
        std::shared_ptr<Vector<T>>& holder,
        const CudaStream& stream) {
    if (vec.location() == target_type) {
        return vec.data();
    }

    holder = Vector<T>::copyFrom(vec, target_type, stream);
    return holder->data();
}

template class HashStrategy<Block<int>>;
template class HashStrategy<Block<float>>;

template class CPUHashStrategy<Block<int>>;
template class CPUHashStrategy<Block<float>>;

template class GPUHashStrategy<Block<int>>;
template class GPUHashStrategy<Block<float>>;
// TODO: Add Block Type Strategies
}  // namespace voxhash