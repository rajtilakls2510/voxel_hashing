
#include "voxhash/core/layer.h"
#include "voxhash/core/block.h"
#include "voxhash/core/voxels.h"

namespace voxhash
{

    template <typename BlockType>
    BlockLayer<BlockType>::BlockLayer(const BlockLayerParams &params) : block_size_(params.block_size), pool_(params.min_allocated_blocks, params.max_allocated_blocks, params.memory_type), type_(params.memory_type)
    {
    }

    template <typename BlockType>
    IndexBlockPair<BlockType> BlockLayer<BlockType>::getBlock(const Index3D &index) const
    {
        const auto it = hash_.find(index);
        return std::make_pair(index, it != hash_.end() ? it->second : nullptr);
    }

    template <typename BlockType>
    ConstIndexBlockPairs<BlockType> BlockLayer<BlockType>::getBlocks(const std::vector<Index3D> &indices) const
    {
        std::vector<Index3D> all_indices;
        std::vector<typename BlockType::Ptr> all_blocks;

        for (const auto &index : indices)
        {
            IndexBlockPair<BlockType> block_pair = getBlock(index);
            all_indices.push_back(block_pair.first);
            all_blocks.push_back(block_pair.second);
        }
        return {all_indices, all_blocks};
    }

    template <typename BlockType>
    bool BlockLayer<BlockType>::allocateBlock(const Index3D &index)
    {
        // First, check if aleady a block exists or not
        IndexBlockPair<BlockType> block_pair = getBlock(index);
        if (block_pair.second)
        {
            // If Block already exists, return True
            return true;
        }
        else
        {
            // Add new_block at hash_[index]
            typename BlockType::Ptr new_block = pool_.popBlock();
            if (!new_block)
                return false;
            hash_[index] = new_block;
        }
        return true;
    }

    template <typename BlockType>
    std::vector<bool> BlockLayer<BlockType>::allocateBlocks(const std::vector<Index3D> &indices)
    {
        std::vector<bool> able_to_allocate;
        able_to_allocate.reserve(indices.size());

        for (const auto &index : indices)
            able_to_allocate.push_back(allocateBlock(index));

        return able_to_allocate;
    }

    template <typename BlockType>
    std::vector<Index3D> BlockLayer<BlockType>::getAllBlockIndices() const
    {
        std::vector<Index3D> indices;
        indices.reserve(hash_.size());

        for (const auto &kv : hash_)
        {
            indices.push_back(kv.first);
        }
        return indices;
    }
    template <typename BlockType>
    std::vector<typename BlockType::Ptr> BlockLayer<BlockType>::getAllBlockPointers() const
    {
        std::vector<typename BlockType::Ptr> ptrs;
        ptrs.reserve(hash_.size());

        for (const auto &kv : hash_)
        {
            ptrs.push_back(kv.second);
        }
        return ptrs;
    }

    template <typename BlockType>
    bool BlockLayer<BlockType>::isBlockAllocated(const Index3D &index) const
    {
        const auto it = hash_.find(index);
        return it != hash_.end();
    }

    template <typename BlockType>
    std::vector<bool> BlockLayer<BlockType>::areBlocksAllocated(const std::vector<Index3D> &indices) const
    {
        std::vector<bool> found;
        found.reserve(indices.size());
        for (const auto &index : indices)
            found.push_back(isBlockAllocated(index));
        return found;
    }

    template <typename BlockType>
    bool BlockLayer<BlockType>::storeBlock(const IndexBlockPair<BlockType> &index_block_pair)
    {
        Index3D index = index_block_pair.first;
        typename BlockType::Ptr block_ptr = index_block_pair.second;
        if (!block_ptr)
            return false;

        bool allocation_success = allocateBlock(index);
        if (!allocation_success)
            return false;

        IndexBlockPair<BlockType> layer_index_block_pair = getBlock(index);
        typename BlockType::Ptr layer_block_ptr = layer_index_block_pair.second;
        if (!layer_block_ptr)
            return false;

        layer_block_ptr->setFrom(*block_ptr);
        return true;
    }

    template <typename BlockType>
    std::vector<bool> BlockLayer<BlockType>::storeBlocks(const ConstIndexBlockPairs<BlockType> &index_block_pairs)
    {

        const auto &indices = index_block_pairs.first;
        const auto &blocks = index_block_pairs.second;

        std::vector<bool> stored;
        stored.reserve(indices.size());
        size_t size = indices.size();

        for (size_t i = 0; i < size; i++)
        {
            IndexBlockPair<BlockType> index_block_pair = {indices[i], blocks[i]};
            stored.push_back(storeBlock(index_block_pair));
        }
        return stored;
    }

    template <typename BlockType>
    bool BlockLayer<BlockType>::deAllocateBlock(const Index3D &index)
    {

        auto it = hash_.find(index);
        if (it == hash_.end()) // Nothing to deallocate
            return true;

        bool deallocated = false;
        // Return block to pool
        if (it->second)
            deallocated = pool_.pushBlock(it->second);

        // Remove from hash
        hash_.erase(it);
        return deallocated;
    }

    template <typename BlockType>
    std::vector<bool> BlockLayer<BlockType>::deAllocateBlocks(const std::vector<Index3D> &indices)
    {
        std::vector<bool> deallocated;
        deallocated.reserve(indices.size());

        for (const auto &index : indices)
            deallocated.push_back(deAllocateBlock(index));
        return deallocated;
    }

    template <typename BlockType>
    BlockLayer<BlockType>::~BlockLayer() {}

    template class BlockLayer<Block<int>>;
    template class BlockLayer<Block<float>>;
    template class BlockLayer<Block<TsdfVoxel>>;
    template class BlockLayer<Block<SemanticVoxel>>;

}