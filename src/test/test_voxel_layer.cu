#include "voxhash/core/layer.h"
#include <iostream>
#include <string>

using namespace voxhash;

using TsdfBlock = Block<TsdfVoxel>;
using TsdfLayer = VoxelBlockLayer<TsdfBlock>;

int main(int argc, char *argv[])
{
    warmupCuda();

    BlockLayerParams p;
    p.block_size = 0.4; // m
    p.memory_type = MemoryType::kDevice;
    p.min_allocated_blocks = 2;
    p.max_allocated_blocks = 5;

    TsdfLayer layer(p);

    Index3D index_to_allocate(10, 1, 2);
    bool allocated = layer.allocateBlock(index_to_allocate);

    std::cout << "Num blocks: " << layer.numBlocks() << " Num voxels: " << layer.numVoxels() << " Voxel size: " << layer.voxel_size() << "\n";

    CudaStreamOwning stream;

    Index3D block_idx(10, 1, 2), voxel_idx(0, 1, 0);
    TsdfVoxel v = layer.getVoxel(block_idx, voxel_idx, stream);
    std::cout << "Tsdf: " << v.tsdf << " Weight: " << v.weight << "\n";

    v.tsdf = 1.0f;
    v.weight = 10.0f;
    layer.storeVoxel(block_idx, voxel_idx, v, stream);

    v = layer.getVoxel(block_idx, voxel_idx, stream);
    std::cout << "Tsdf: " << v.tsdf << " Weight: " << v.weight << "\n";

    // ============

    Vector3f positionL(1.1, 2.1, 3.2);
    getBlockAndVoxelIndexFromPosition(layer.block_size(), positionL, &block_idx, &voxel_idx);
    allocated = layer.allocateBlock(block_idx);

    v = layer.getVoxel(positionL, stream);
    std::cout << "Tsdf: " << v.tsdf << " Weight: " << v.weight << "\n";

    v.tsdf = 2.0f;
    v.weight = 15.0f;
    layer.storeVoxel(positionL, v, stream);

    v = layer.getVoxel(positionL, stream);
    std::cout << "Tsdf: " << v.tsdf << " Weight: " << v.weight << "\n";

    // ===========

    std::vector<Index3D> block_indices;
    std::vector<Index3D> voxel_indices;
    block_indices.push_back(Index3D(0, 0, 1));
    block_indices.push_back(Index3D(-1, 0, 0));
    block_indices.push_back(Index3D(0, -1, 0));
    voxel_indices.push_back(Index3D(2, 0, 1));
    voxel_indices.push_back(Index3D(2, 1, 0));
    voxel_indices.push_back(Index3D(2, 0, 0));
    std::vector<TsdfVoxel> voxels_to_store;
    voxels_to_store.emplace_back(-1.0f, 10.0f);
    voxels_to_store.emplace_back(1.0f, 20.0f);
    voxels_to_store.emplace_back(-1.5f, 30.0f);

    layer.allocateBlocks(block_indices);

    layer.getBlock(block_indices[0]).second->setVoxel(voxel_indices[0], voxels_to_store[0], stream);
    // layer.getBlock(block_indices[1]).second->setVoxel(voxel_indices[1], voxels_to_store[1], stream);
    layer.getBlock(block_indices[2]).second->setVoxel(voxel_indices[2], voxels_to_store[2], stream);

    IndexPairs block_and_voxel_indices = {voxel_indices, block_indices};
    Vector<TsdfVoxel> voxels = layer.getVoxels(block_and_voxel_indices, stream);
    Vector<TsdfVoxel>::Ptr voxels_host = Vector<TsdfVoxel>::copyFrom(voxels, MemoryType::kHost, stream);

    std::cout << "Voxels: \n";
    for (size_t i = 0; i < voxels_host->size(); i++)
        std::cout << "(" << voxels_host->data()[i].tsdf << "," << voxels_host->data()[i].weight << ")\n";

    return 0;
}