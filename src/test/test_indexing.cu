#include "voxhash/core/indexing.h"
#include "voxhash/core/vector.h"
#include <iostream>
#include <string>
#include <sstream>

using namespace voxhash;

std::string idxToStr(const Index3D &idx)
{
    std::ostringstream oss;
    oss << "(" << idx.x << ", " << idx.y << ", " << idx.z << ")";
    return oss.str();
}

std::string idxToStr(const Vector3f &idx)
{
    std::ostringstream oss;
    oss << "(" << idx.x << ", " << idx.y << ", " << idx.z << ")";
    return oss.str();
}

int main(int argc, char *argv[])
{
    const float block_size = 0.4; // m
    std::cout << "Voxel size: " << blockSizeToVoxelSize(block_size) << " Block size: " << voxelSizeToBlockSize(blockSizeToVoxelSize(block_size)) << "\n";

    Vector3f test_position(1.0f, 2.0f, 3.0f);
    std::cout << "Block Position: " << idxToStr(test_position) << " Block Idx: " << idxToStr(getBlockIndexFromPosition(block_size, test_position)) << "\n";

    Index3D test_block_idx(2, 5, 7);
    std::cout << "Block idx: " << idxToStr(test_block_idx) << " Block Position: " << idxToStr(getPositionFromBlockIndex(block_size, test_block_idx)) << "\n";
    std::cout << "Block idx: " << idxToStr(test_block_idx) << " Block Center Position: " << idxToStr(getCenterPositionFromBlockIndex(block_size, test_block_idx)) << "\n";

    Index3D block_idx(0, 0, 0), voxel_idx(0, 0, 0);
    getBlockAndVoxelIndexFromPosition(block_size, test_position, &block_idx, &voxel_idx);
    std::cout << "Position: " << idxToStr(test_position) << " Block idx: " << idxToStr(block_idx) << " Voxel idx: " << idxToStr(voxel_idx) << "\n";

    Vector3f position_ = getPositionFromBlockAndVoxelIndex(block_size, block_idx, voxel_idx);
    std::cout << "Position: " << idxToStr(test_position) << " Block idx: " << idxToStr(block_idx) << " Voxel idx: " << idxToStr(voxel_idx) << " Received pos: " << idxToStr(position_) << "\n";

    Vector3f center_position_ = getCenterPositionFromBlockAndVoxelIndex(block_size, block_idx, voxel_idx);
    std::cout << "Position: " << idxToStr(test_position) << " Block idx: " << idxToStr(block_idx) << " Voxel idx: " << idxToStr(voxel_idx) << " Received Center pos: " << idxToStr(center_position_) << "\n";

    std::vector<Vector3f> positions_to_test;
    positions_to_test.push_back(Vector3f(1.0f, 2.0f, 3.025f));
    positions_to_test.push_back(Vector3f(1.0f, 2.2f, 3.5f));
    positions_to_test.push_back(Vector3f(0.5f, 2.2f, 4.0f));

    CudaStreamOwning cuda_stream;
    Vector<Vector3f>::Ptr positions_vec = Vector<Vector3f>::copyFrom(positions_to_test, MemoryType::kDevice, cuda_stream);

    std::pair<Vector<Index3D>, Vector<Index3D>> block_and_voxel_indices = getBlockAndVoxelIndicesFromPositions(block_size, *positions_vec);
    Vector<Index3D>::Ptr block_indices = Vector<Index3D>::copyFrom(block_and_voxel_indices.first, MemoryType::kHost, cuda_stream);
    Vector<Index3D>::Ptr voxel_indices = Vector<Index3D>::copyFrom(block_and_voxel_indices.second, MemoryType::kHost, cuda_stream);

    for (size_t i = 0; i < positions_to_test.size(); i++)
        std::cout << "Position: " << idxToStr(positions_to_test[i]) << " Block idx: " << idxToStr(block_indices->data()[i]) << " Voxel idx: " << idxToStr(voxel_indices->data()[i]) << "\n";

    return 0;
}