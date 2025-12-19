#pragma once

#include "voxhash/core/vector.h"

namespace voxhash
{

    template <typename VoxelType>
    class Block : public Vector<VoxelType>
    {
    public:
        using Ptr = std::shared_ptr<Block<VoxelType>>;
        static constexpr size_t kVoxelsPerSide = 8;
        static constexpr size_t kNumVoxels =
            kVoxelsPerSide * kVoxelsPerSide * kVoxelsPerSide;

        Block(MemoryType type);
        void clear(const CudaStream &stream = CudaStreamOwning()) override;

        VoxelType getVoxel(const Index3D &index, const CudaStream &cuda_stream = CudaStreamOwning()) const;
        void setVoxel(const Index3D &index, const VoxelType value, const CudaStream &cuda_stream = CudaStreamOwning());

        static std::shared_ptr<Block<VoxelType>> copyFrom(const Block<VoxelType> &src, MemoryType target_type,
                                                          const CudaStream &stream = CudaStreamOwning());

    protected:
        size_t idx(const Index3D &index) const;
    };
} // namespace voxhash