#pragma once

#include <unordered_map>
#include <utility>
#include <vector>

#include "voxhash/core/types.h"
#include "voxhash/core/vector.h"

namespace voxhash {

// Taken from Nvblox
struct Index3DHash {
    /// number was arbitrarily chosen with no good justification
    static constexpr size_t sl = 17191;
    static constexpr size_t sl2 = sl * sl;

    std::size_t operator()(const Index3D& index) const {
        return static_cast<size_t>(index.x + index.y * sl + index.z * sl2);
    }
};

template <typename BlockType>
using Index3DHashMapType = std::unordered_map<Index3D, typename BlockType::Ptr, Index3DHash>;

template <typename BlockType>
class HashStrategy {
    // Interface to provide hashing functionality
public:
    explicit HashStrategy(
            std::shared_ptr<CudaStream> cuda_stream = std::make_shared<CudaStreamOwning>())
        : stream_(std::move(cuda_stream)) {}
    virtual ~HashStrategy() = default;

    virtual bool findValue(const Index3D key, typename BlockType::Ptr& value) const = 0;
    virtual void findValues(
            const std::vector<Index3D> keys,
            std::vector<typename BlockType::Ptr>& values,
            std::vector<bool>& found) const = 0;
    virtual std::vector<Index3D> getAllKeys() const = 0;
    virtual std::vector<typename BlockType::Ptr> getAllValues() const = 0;
    virtual bool insertValue(const Index3D key, const typename BlockType::Ptr value) = 0;
    virtual void insertValues(
            const std::vector<Index3D> keys,
            const std::vector<typename BlockType::Ptr> values,
            std::vector<bool>& inserted) = 0;
    virtual bool eraseValue(const Index3D key) = 0;
    virtual void eraseValues(const std::vector<Index3D> keys, std::vector<bool>& erased) = 0;
    virtual size_t size() const = 0;

protected:
    std::shared_ptr<CudaStream> stream_;
};

template <typename BlockType>
class CPUHashStrategy : public HashStrategy<BlockType> {
public:
    CPUHashStrategy() : HashStrategy<BlockType>() {}
    virtual ~CPUHashStrategy() {}

    virtual bool findValue(const Index3D key, typename BlockType::Ptr& value) const override;
    virtual void findValues(
            const std::vector<Index3D> keys,
            std::vector<typename BlockType::Ptr>& values,
            std::vector<bool>& found) const override;
    virtual std::vector<Index3D> getAllKeys() const;
    virtual std::vector<typename BlockType::Ptr> getAllValues() const;
    virtual bool insertValue(const Index3D key, const typename BlockType::Ptr value) override;
    virtual void insertValues(
            const std::vector<Index3D> keys,
            const std::vector<typename BlockType::Ptr> values,
            std::vector<bool>& inserted);
    virtual bool eraseValue(const Index3D key) override;
    virtual void eraseValues(const std::vector<Index3D> keys, std::vector<bool>& erased) override;
    virtual size_t size() const override { return hash_.size(); }

private:
    Index3DHashMapType<BlockType> hash_;
};

// class

}  // namespace voxhash