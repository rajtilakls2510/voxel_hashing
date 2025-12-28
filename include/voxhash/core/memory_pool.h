#pragma once

#include <stack>

#include "voxhash/core/block.h"
#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/voxels.h"

namespace voxhash {

// Stores free pre-allocated blocks to be used
template <typename BlockType>
class BlockMemoryPool {
public:
    BlockMemoryPool(size_t min_allocated_blocks, size_t max_allocated_blocks, MemoryType type);
    virtual ~BlockMemoryPool();
    size_t size() const;
    typename BlockType::Ptr popBlock();
    bool pushBlock(typename BlockType::Ptr block);

protected:
    size_t min_allocated_blocks_{4}, max_allocated_blocks_{64};
    MemoryType type_{MemoryType::kHost};
    std::stack<typename BlockType::Ptr> blocks_;
    virtual void ensureCapacity();
};

}  // namespace voxhash