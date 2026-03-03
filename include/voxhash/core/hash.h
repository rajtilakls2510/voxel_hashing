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
    explicit HashStrategy() {}
    virtual ~HashStrategy() = default;

    virtual std::pair<Vector<Bool>, Vector<typename BlockType::VoxelType*>> findValues(
            const Vector<Index3D>& keys, const CudaStream& stream = CudaStreamOwning()) const = 0;
    virtual std::pair<Vector<Index3D>, Vector<typename BlockType::VoxelType*>> getAllKeyValues(
            const CudaStream& stream = CudaStreamOwning()) const = 0;
    virtual Vector<Bool> insertValues(
            const Vector<Index3D>& keys,
            const Vector<typename BlockType::VoxelType*>& values,
            const CudaStream& stream = CudaStreamOwning()) = 0;
    virtual Vector<Bool> eraseValues(
            const Vector<Index3D>& keys, const CudaStream& stream = CudaStreamOwning()) = 0;
    virtual size_t size() const = 0;
};

template <typename BlockType>
class CPUHashStrategy : public HashStrategy<BlockType> {
public:
    CPUHashStrategy(MemoryType type = MemoryType::kHost) : HashStrategy<BlockType>(), type_(type) {}
    virtual ~CPUHashStrategy() {}

    virtual std::pair<Vector<Bool>, Vector<typename BlockType::VoxelType*>> findValues(
            const Vector<Index3D>& keys,
            const CudaStream& stream = CudaStreamOwning()) const override;
    virtual Vector<Bool> insertValues(
            const Vector<Index3D>& keys,
            const Vector<typename BlockType::VoxelType*>& values,
            const CudaStream& stream = CudaStreamOwning()) override;
    virtual std::pair<Vector<Index3D>, Vector<typename BlockType::VoxelType*>> getAllKeyValues(
            const CudaStream& stream = CudaStreamOwning()) const override;
    virtual Vector<Bool> eraseValues(
            const Vector<Index3D>& keys, const CudaStream& stream = CudaStreamOwning()) override;
    virtual size_t size() const override { return hash_.size(); }

protected:
    CPUHashMapType<BlockType> hash_;
    MemoryType type_{MemoryType::kHost};
};

template <typename BlockType>
class GPUHashStrategy : public HashStrategy<BlockType> {
public:
    GPUHashStrategy(size_t num_increase_objects = 1024, MemoryType type = MemoryType::kDevice);
    virtual ~GPUHashStrategy() override;
    virtual std::pair<Vector<Bool>, Vector<typename BlockType::VoxelType*>> findValues(
            const Vector<Index3D>& keys, const CudaStream& stream = CudaStreamOwning()) const;
    virtual Vector<Bool> insertValues(
            const Vector<Index3D>& keys,
            const Vector<typename BlockType::VoxelType*>& values,
            const CudaStream& stream = CudaStreamOwning());
    virtual std::pair<Vector<Index3D>, Vector<typename BlockType::VoxelType*>> getAllKeyValues(
            const CudaStream& stream = CudaStreamOwning()) const;
    virtual Vector<Bool> eraseValues(
            const Vector<Index3D>& keys, const CudaStream& stream = CudaStreamOwning()) override;
    virtual size_t size() const override { return hash_.size(); }

protected:
    size_t num_increase_objects_, current_objects_;
    GPUHashMapType<BlockType> hash_;
    MemoryType type_{MemoryType::kDevice};

    void recreateHash(size_t increase_factor, const CudaStream& stream = CudaStreamOwning());
};

template <typename T>
T* optionallyCopyDataPtr(
        const Vector<T>& vec,
        MemoryType target_type,
        std::shared_ptr<Vector<T>>& holder,
        const CudaStream& stream = CudaStreamOwning());

}  // namespace voxhash