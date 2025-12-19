
#include "voxhash/core/block.h"
#include "voxhash/core/voxels.h"

namespace voxhash
{

    template <typename VoxelType>
    Block<VoxelType>::Block(MemoryType type) : Vector<VoxelType>(kNumVoxels, type)
    {
    }

    template <typename VoxelType>
    std::shared_ptr<Block<VoxelType>> Block<VoxelType>::copyFrom(const Block<VoxelType> &src, MemoryType target_type,
                                                                 const CudaStream &stream)
    {
        auto dst = std::make_shared<Block<VoxelType>>(target_type);

        checkCudaErrors(
            cudaMemcpyAsync(dst->data(),
                            src.data(),
                            sizeof(VoxelType) * kNumVoxels,
                            cudaMemcpyDefault,
                            stream));

        return dst;
    }

    template <typename VoxelType>
    VoxelType Block<VoxelType>::getVoxel(const Index3D &index, const CudaStream &cuda_stream) const
    {
        VoxelType v;
        checkCudaErrors(cudaMemcpyAsync(&v, this->ptr_ + idx(index), sizeof(VoxelType), cudaMemcpyDefault, cuda_stream));
        return v;
    }

    template <typename VoxelType>
    void Block<VoxelType>::setVoxel(const Index3D &index, const VoxelType value, const CudaStream &cuda_stream)
    {
        checkCudaErrors(cudaMemcpyAsync(this->ptr_ + idx(index), &value, sizeof(VoxelType), cudaMemcpyDefault, cuda_stream));
    }

    template <typename VoxelType>
    size_t Block<VoxelType>::idx(const Index3D &index) const
    {
        return index.x * kVoxelsPerSide * kVoxelsPerSide + index.y * kVoxelsPerSide + index.z;
    }

    template <typename VoxelType>
    void Block<VoxelType>::clear(const CudaStream &cuda_stream)
    {
        checkCudaErrors(cudaMemsetAsync(this->ptr_, 0, sizeof(VoxelType) * kNumVoxels, cuda_stream));
    }

    template class Block<int>;
    template class Block<float>;
    template class Block<TsdfVoxel>;
    template class Block<SemanticVoxel>;
}
