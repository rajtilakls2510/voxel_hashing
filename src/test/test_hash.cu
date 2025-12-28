#include <iostream>
#include <memory>

#include "voxhash/core/block.h"
#include "voxhash/core/hash.h"

using namespace voxhash;

using FloatBlock = Block<float>;

int main(int argc, char* argv[]) {
    MemoryType host_type = MemoryType::kHost;
    MemoryType device_type = MemoryType::kDevice;
    IndexBlockPair<FloatBlock> p = {Index3D(0, 1, 0), std::make_shared<FloatBlock>(host_type)};

    std::vector<Index3D> v = {Index3D(0, 0, 1), Index3D(1, 0, 1)};
    std::vector<FloatBlock::Ptr> b = {
            std::make_shared<FloatBlock>(host_type), std::make_shared<FloatBlock>(device_type)};
    FloatBlock::Ptr some_other_block = std::make_shared<FloatBlock>(device_type);

    ConstIndexBlockPairs<FloatBlock> pp = {v, b};

    // pp.second.push_back(some_other_block);

    FloatBlock::Ptr bl = pp.second[1];
    float test_voxel_value = 100;
    Index3D index_to_set(0, 0, 1);
    bl->setVoxel(index_to_set, test_voxel_value);

    return 0;
}
