#include <iostream>

#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/layer.h"

using namespace voxhash;
using TsdfBlock = Block<TsdfVoxel>;
using TsdfLayer = BlockLayer<TsdfBlock>;

void printAllocations(ConstIndexBlockPairs<TsdfBlock>& index_block_pairs) {
    const auto& indices = index_block_pairs.first;
    const auto& blocks = index_block_pairs.second;

    const size_t n = std::min(indices.size(), blocks.size());

    for (size_t i = 0; i < n; ++i) {
        const Index3D& idx = indices[i];

        std::cout << "(" << idx.x << ", " << idx.y << ", " << idx.z << "): ";

        if (blocks[i])
            std::cout << "allocated\n";
        else
            std::cout << "not allocated\n";
    }
}

void printAllocations(std::vector<Index3D>& indices, std::vector<bool>& able_to_allocate) {
    const size_t n = std::min(indices.size(), able_to_allocate.size());
    for (size_t i = 0; i < n; ++i) {
        const Index3D& idx = indices[i];

        std::cout << "(" << idx.x << ", " << idx.y << ", " << idx.z << "): ";

        if (able_to_allocate[i])
            std::cout << "allocated\n";
        else
            std::cout << "not allocated\n";
    }
}

void printIndices(const std::vector<Index3D>& indices) {
    std::cout << "Indices: ";
    for (const auto& index : indices) {
        std::cout << "(" << index.x << "," << index.y << "," << index.z << "), ";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    warmupCuda();

    BlockLayerParams p;
    p.block_size = 0.4;  // m
    p.memory_type = MemoryType::kDevice;
    p.min_allocated_blocks = 2;
    p.max_allocated_blocks = 5;

    TsdfLayer layer(p);

    Index3D index_to_allocate(10, 1, 2);
    IndexBlockPair<TsdfBlock> block_pair = layer.getBlock(index_to_allocate);

    if (block_pair.second)
        std::cout << "Block exists\n";
    else
        std::cout << "Block does not exist\n";

    bool allocated = layer.allocateBlock(index_to_allocate);
    if (allocated) {
        block_pair = layer.getBlock(index_to_allocate);
        if (block_pair.second)
            std::cout << "Block exists\n";
        else
            std::cout << "Block does not exist\n";
    } else {
        std::cout << "Couldn't allocate block\n";
    }

    std::vector<Index3D> indices_to_check;
    indices_to_check.emplace_back(0, 1, 0);
    indices_to_check.emplace_back(1, 0, 0);
    indices_to_check.emplace_back(0, 0, 1);

    ConstIndexBlockPairs<TsdfBlock> index_block_pairs = layer.getBlocks(indices_to_check);
    printAllocations(index_block_pairs);

    std::vector<Index3D> indices_to_allocate;
    indices_to_allocate.emplace_back(0, 1, 0);
    indices_to_allocate.emplace_back(0, 0, 1);

    std::vector<bool> able_to_allocate = layer.allocateBlocks(indices_to_allocate);
    printAllocations(indices_to_allocate, able_to_allocate);

    ConstIndexBlockPairs<TsdfBlock> index_block_pairs2 = layer.getBlocks(indices_to_check);
    printAllocations(index_block_pairs2);

    std::vector<Index3D> all_block_indices = layer.getAllBlockIndices();
    std::vector<TsdfBlock::Ptr> all_block_ptrs = layer.getAllBlockPointers();
    printIndices(all_block_indices);

    std::vector<bool> are_allocated = layer.areBlocksAllocated(indices_to_check);
    printAllocations(indices_to_check, are_allocated);

    TsdfBlock::Ptr block_to_store = std::make_shared<TsdfBlock>(MemoryType::kHost);
    TsdfVoxel vox = {1.0f, 10.0f};
    Index3D index_to_set(1, 0, 0);
    block_to_store->setVoxel(index_to_set, vox);
    IndexBlockPair<TsdfBlock> ibp = {index_to_set, block_to_store};
    bool is_set = layer.storeBlock(ibp);
    if (!is_set)
        std::cout << "Unable to set\n";
    else {
        std::cout << "Able to set\n";
        std::vector<bool> are_allocated = layer.areBlocksAllocated(indices_to_check);
        printAllocations(indices_to_check, are_allocated);
    }

    layer.deAllocateBlock(Index3D(0, 1, 0));
    all_block_indices = layer.getAllBlockIndices();
    printIndices(all_block_indices);

    return 0;
}