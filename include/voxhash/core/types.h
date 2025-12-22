#pragma once

#include <string>
#include <vector>

namespace voxhash
{

    enum class MemoryType
    {
        kHost,
        kDevice,
        kUnified
    };

    inline std::string to_string(MemoryType type)
    {
        switch (type)
        {
        case MemoryType::kHost:
            return "Host";
        case MemoryType::kDevice:
            return "Device";
        case MemoryType::kUnified:
            return "Unified";
        default:
            return "Unknown Device";
        }
    }

    using Bool = uint8_t;

    struct Index3D
    {
        int x, y, z;
        __host__ __device__ Index3D(int x, int y, int z) : x(x), y(y), z(z) {}
        __host__ __device__ bool operator==(const Index3D &other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

    struct Vector3f
    {
        float x, y, z;
        __host__ __device__ Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
    };

    template <typename BlockType>
    using IndexBlockPair = std::pair<Index3D, typename BlockType::Ptr>;

    // Note vector<Index3D> and vector<Block pointer> modification is removed. Modification of the content of each block is not removed.
    template <typename BlockType>
    using ConstIndexBlockPairs = std::pair<const std::vector<Index3D>, const std::vector<typename BlockType::Ptr>>;

    using IndexPairs = std::pair<const std::vector<Index3D>, const std::vector<Index3D>>;

}