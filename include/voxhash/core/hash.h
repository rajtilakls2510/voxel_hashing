#pragma once

#include "voxhash/core/types.h"
#include "voxhash/core/vector.h"
#include <unordered_map>
#include <utility>
#include <vector>

namespace voxhash
{

    template <typename BlockType>
    using IndexBlockPair = std::pair<Index3D, typename BlockType::Ptr>;

    // Note vector<Index3D> and vector<Block pointer> modification is removed. Modification of the content of each block is not removed.
    template <typename BlockType>
    using ConstIndexBlockPairs = std::pair<const std::vector<Index3D>, const std::vector<typename BlockType::Ptr>>;

    // Taken from Nvblox
    struct Index3DHash
    {
        /// number was arbitrarily chosen with no good justification
        static constexpr size_t sl = 17191;
        static constexpr size_t sl2 = sl * sl;

        std::size_t operator()(const Index3D &index) const
        {
            return static_cast<size_t>(index.x + index.y * sl + index.z * sl2);
        }
    };

    template <typename BlockType>
    using Index3DHashMapType =
        std::unordered_map<Index3D, typename BlockType::Ptr, Index3DHash>;

} // namespace voxhash