#include "voxhash/core/cuda_utils.h"
#include "voxhash/hashing/memory_pool.h"
#include "voxhash/core/voxels.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>
using namespace std::chrono;

using namespace voxhash;
using TsdfBlock = Block<TsdfVoxel>;

int main(int argc, char *argv[])
{
    warmupCuda();
    MemoryType type = MemoryType::kDevice;

    size_t min_alloc = 2;
    size_t max_alloc = 5;

    size_t to_pull = 8;
    std::vector<TsdfBlock::Ptr> pulled_blocks;

    {
        BlockMemoryPool<TsdfBlock> pool(min_alloc, max_alloc, type);

        for (size_t i = 0; i < to_pull; i++)
        {
            std::cout << "Pulling...\n";
            pulled_blocks.emplace_back(pool.popBlock());
            std::cout << "Size of pool: " << pool.size() << "\n";
        }

        for (size_t i = 0; i < pulled_blocks.size(); i++)
        {
            std::cout << "Pushing...\n";

            pool.pushBlock(std::move(pulled_blocks[i]));
            std::cout << "Size of pool: " << pool.size() << "\n";
        }
    }

    return 0;
}