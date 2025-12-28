#pragma once
#include <memory>

namespace voxhash {

struct TsdfVoxel {
    float tsdf{0.0f};
    float weight{0.0f};
    __host__ __device__ TsdfVoxel() {}
    __host__ __device__ TsdfVoxel(float tsdf, float weight) : tsdf(tsdf), weight(weight) {}
};

struct SemanticVoxel {
    uint16_t label{0};
    double weight{0.0};
    __host__ __device__ SemanticVoxel() {}
    __host__ __device__ SemanticVoxel(uint16_t label, double weight)
        : label(label), weight(weight) {}
};

}  // namespace voxhash