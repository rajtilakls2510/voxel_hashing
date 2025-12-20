#pragma once

#include "voxhash/core/vector.h"

namespace voxhash
{

    template <typename _VoxelType>
    class Block : public Vector<_VoxelType>
    {
    public:
        using Ptr = std::shared_ptr<Block<_VoxelType>>;
        using VoxelType = _VoxelType;

        static constexpr size_t kVoxelsPerSide = 8;
        static constexpr size_t kNumVoxels =
            kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;

        Block(MemoryType type);
        void clear(const CudaStream &stream = CudaStreamOwning()) override;

        _VoxelType getVoxel(const Index3D &index, const CudaStream &cuda_stream = CudaStreamOwning()) const;
        void setVoxel(const Index3D &index, const _VoxelType value, const CudaStream &cuda_stream = CudaStreamOwning());

        static std::shared_ptr<Block<_VoxelType>> copyFrom(const Block<_VoxelType> &src, MemoryType target_type,
                                                           const CudaStream &stream = CudaStreamOwning());

        __host__ __device__ static size_t idx(const Index3D &index);
    };
} // namespace voxhash