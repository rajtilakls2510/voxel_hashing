#pragma once

#include <string>

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

    struct Index3D
    {
        int x, y, z;
        Index3D(int x, int y, int z) : x(x), y(y), z(z) {}
        bool operator==(const Index3D &other) const noexcept
        {
            return x == other.x && y == other.y && z == other.z;
        }
    };

}