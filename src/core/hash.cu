#include <voxhash/core/block.h>
#include <voxhash/core/hash.h>

namespace voxhash {

template <typename BlockType>
bool CPUHashStrategy<BlockType>::findValue(
        const Index3D key, typename BlockType::Ptr& value) const {
    const auto it = hash_.find(key);
    if (it == hash_.end()) return false;
    value = it->second;
    return true;
}

template <typename BlockType>
void CPUHashStrategy<BlockType>::findValues(
        const std::vector<Index3D> keys,
        std::vector<typename BlockType::Ptr>& values,
        std::vector<bool>& found) const {
    found.resize(keys.size());
    values.resize(keys.size());

    for (size_t idx = 0; idx < keys.size(); idx++) found[idx] = findValue(keys[idx], values[idx]);
}

template <typename BlockType>
std::vector<Index3D> CPUHashStrategy<BlockType>::getAllKeys() const {
    std::vector<Index3D> indices;
    indices.reserve(hash_.size());

    for (const auto& kv : hash_) {
        indices.push_back(kv.first);
    }
    return indices;
}

template <typename BlockType>
std::vector<typename BlockType::Ptr> CPUHashStrategy<BlockType>::getAllValues() const {
    std::vector<typename BlockType::Ptr> ptrs;
    ptrs.reserve(hash_.size());

    for (const auto& kv : hash_) {
        ptrs.push_back(kv.second);
    }
    return ptrs;
}

template <typename BlockType>
bool CPUHashStrategy<BlockType>::insertValue(
        const Index3D key, const typename BlockType::Ptr value) {
    const auto it = hash_.find(key);
    if (it != hash_.end())  // Reject insertion if a value already exists
        return value == it->second;
    hash_[key] = value;
    return true;
}

template <typename BlockType>
void CPUHashStrategy<BlockType>::insertValues(
        const std::vector<Index3D> keys,
        const std::vector<typename BlockType::Ptr> values,
        std::vector<bool>& inserted) {
    inserted.resize(keys.size());

    for (size_t idx = 0; idx < keys.size(); idx++)
        inserted[idx] = insertValue(keys[idx], values[idx]);
}

template <typename BlockType>
bool CPUHashStrategy<BlockType>::eraseValue(const Index3D key) {
    const auto it = hash_.find(key);
    if (it == hash_.end()) return true;
    hash_.erase(it);
    return true;
}

template <typename BlockType>
void CPUHashStrategy<BlockType>::eraseValues(
        const std::vector<Index3D> keys, std::vector<bool>& erased) {
    erased.resize(keys.size());

    for (size_t idx = 0; idx < keys.size(); idx++) erased[idx] = eraseValue(keys[idx]);
}

template class HashStrategy<Block<int>>;
template class HashStrategy<Block<float>>;

template class CPUHashStrategy<Block<int>>;
template class CPUHashStrategy<Block<float>>;
}  // namespace voxhash