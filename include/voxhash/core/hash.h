#pragma once

#include <stdgpu/iterator.h>  // device_begin, device_end
#include <stdgpu/memory.h>    // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>  // STDGPU_HOST_DEVICE
#include <thrust/transform.h>

#include <stdgpu/unordered_map.cuh>  // stdgpu::unordered_map
#include <unordered_map>
#include <utility>
#include <vector>

#include "voxhash/core/types.h"
#include "voxhash/core/vector.h"

namespace voxhash {

// Taken from Nvblox
struct Index3DHash {
    /// number was arbitrarily chosen with no good justification
    static constexpr size_t sl = 17191;
    static constexpr size_t sl2 = sl * sl;

    __host__ __device__ std::size_t operator()(const Index3D& index) const {
        return static_cast<size_t>(index.x + index.y * sl + index.z * sl2);
    }
};

template <typename BlockType>
using Index3DHashMapType = std::unordered_map<Index3D, typename BlockType::Ptr, Index3DHash>;
template <typename BlockType>
using CPUHashMapType = std::unordered_map<Index3D, typename BlockType::VoxelType*, Index3DHash>;
template <typename BlockType>
using GPUHashMapType = stdgpu::unordered_map<Index3D, typename BlockType::VoxelType*, Index3DHash>;

template <typename BlockType>
class HashStrategy {
    // Interface to provide hashing functionality
public:
    explicit HashStrategy(
            std::shared_ptr<CudaStream> cuda_stream = std::make_shared<CudaStreamOwning>())
        : stream_(std::move(cuda_stream)) {}
    virtual ~HashStrategy() = default;

    virtual std::pair<Vector<Bool>, Vector<typename BlockType::VoxelType*>> findValues(
            const Vector<Index3D>& keys) const = 0;
    virtual std::pair<Vector<Index3D>, Vector<typename BlockType::VoxelType*>> getAllKeyValues()
            const = 0;
    virtual Vector<Bool> insertValues(
            const Vector<Index3D>& keys, const Vector<typename BlockType::VoxelType*>& values) = 0;
    virtual Vector<Bool> eraseValues(const Vector<Index3D>& keys) = 0;
    virtual size_t size() const = 0;

protected:
    std::shared_ptr<CudaStream> stream_;
};

// template <typename BlockType>
// class CPUHashStrategy : public HashStrategy<BlockType> {
// public:
//     CPUHashStrategy() : HashStrategy<BlockType>() {}
//     virtual ~CPUHashStrategy() {}

//     virtual bool findValue(const Index3D key, typename BlockType::Ptr& value) const override;
//     virtual void findValues(
//             const std::vector<Index3D> keys,
//             std::vector<typename BlockType::Ptr>& values,
//             std::vector<bool>& found) const override;
//     virtual std::vector<Index3D> getAllKeys() const;
//     virtual std::vector<typename BlockType::Ptr> getAllValues() const;
//     virtual bool insertValue(const Index3D key, const typename BlockType::Ptr value) override;
//     virtual void insertValues(
//             const std::vector<Index3D> keys,
//             const std::vector<typename BlockType::Ptr> values,
//             std::vector<bool>& inserted);
//     virtual bool eraseValue(const Index3D key) override;
//     virtual void eraseValues(const std::vector<Index3D> keys, std::vector<bool>& erased)
//     override; virtual size_t size() const override { return hash_.size(); }

// protected:
//     Index3DHashMapType<BlockType> hash_;
// };

template <typename BlockType>
class GPUHashStrategy : public HashStrategy<BlockType> {
public:
    GPUHashStrategy(
            std::shared_ptr<CudaStream> cuda_stream = std::make_shared<CudaStreamOwning>(),
            size_t num_increase_objects = 1024,
            MemoryType type = MemoryType::kDevice);
    virtual ~GPUHashStrategy() override;
    virtual std::pair<Vector<Bool>, Vector<typename BlockType::VoxelType*>> findValues(
            const Vector<Index3D>& keys) const;
    virtual Vector<Bool> insertValues(
            const Vector<Index3D>& keys, const Vector<typename BlockType::VoxelType*>& values);
    virtual std::pair<Vector<Index3D>, Vector<typename BlockType::VoxelType*>> getAllKeyValues()
            const;
    virtual Vector<Bool> eraseValues(const Vector<Index3D>& keys) override;
    virtual size_t size() const override { return hash_.size(); }

protected:
    size_t num_increase_objects_, current_objects_;
    GPUHashMapType<BlockType> hash_;
    MemoryType type_{MemoryType::kDevice};

    void recreateHash(size_t increase_factor);
};

}  // namespace voxhash