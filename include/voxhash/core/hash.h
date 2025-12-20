#pragma once

#include "voxhash/core/types.h"
#include "voxhash/core/vector.h"
#include <unordered_map>
#include <utility>
#include <vector>

namespace voxhash
{

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