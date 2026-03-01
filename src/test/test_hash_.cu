#include <gtest/gtest.h>
#include <voxhash/core/block.h>
#include <voxhash/core/cuda_utils.h>
#include <voxhash/core/hash.h>

#include <memory>

using namespace voxhash;
using FloatBlock = Block<float>;

TEST(TestCpuHash, FindOneInsertOneFindOne) {
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<CPUHashStrategy<FloatBlock>>();

    // Search for a non-existing value in the hash
    Index3D index_to_search(0, 1, 0);
    FloatBlock::Ptr dummy_value = nullptr;
    EXPECT_FALSE(hash->findValue(index_to_search, dummy_value));

    // Insert a value
    FloatBlock::Ptr value_to_insert = std::make_shared<FloatBlock>(MemoryType::kHost);
    Index3D voxel_idx_to_set(1, 0, 0);
    value_to_insert->setVoxel(voxel_idx_to_set, 5.0);

    EXPECT_TRUE(hash->insertValue(index_to_search, value_to_insert));

    // Search for the recently inserted value
    EXPECT_TRUE(hash->findValue(index_to_search, dummy_value));
    EXPECT_DOUBLE_EQ(dummy_value->getVoxel(voxel_idx_to_set), 5.0);

    // Insertion to the same key but different values should be rejected
    value_to_insert = nullptr;
    EXPECT_FALSE(hash->insertValue(index_to_search, value_to_insert));
}

TEST(TestCpuHash, InsertMultipleFindValues) {
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<CPUHashStrategy<FloatBlock>>();

    std::vector<Index3D> block_indices_to_insert = {Index3D(0, 1, 0), Index3D(1, 0, 0)};
    std::vector<Index3D> block_indices_to_find = {
            Index3D(0, 1, 0), Index3D(1, 0, 0), Index3D(0, 0, 1)};

    for (size_t i = 0; i < block_indices_to_insert.size(); i++) {
        FloatBlock::Ptr value_to_insert = std::make_shared<FloatBlock>(MemoryType::kHost);
        value_to_insert->setVoxel(Index3D(0, 1, 0), i + 1);
        EXPECT_TRUE(hash->insertValue(block_indices_to_insert[i], value_to_insert));
    }

    std::vector<bool> found;
    std::vector<FloatBlock::Ptr> values;
    hash->findValues(block_indices_to_find, values, found);
    for (size_t i = 0; i < block_indices_to_find.size(); i++) {
        if (i < 2) {
            EXPECT_TRUE(found[i]);
            EXPECT_DOUBLE_EQ(values[i]->getVoxel(Index3D(0, 1, 0)), i + 1);
        } else {
            EXPECT_FALSE(found[i]);
        }
    }
}

TEST(TestCpuHash, InsertOneInsertManyFindMany) {
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<CPUHashStrategy<FloatBlock>>();

    // Insert one
    Index3D first_idx(1, 0, 0);
    FloatBlock::Ptr value_to_insert = std::make_shared<FloatBlock>(MemoryType::kHost);
    value_to_insert->setVoxel(Index3D(0, 1, 0), 10.0);
    EXPECT_TRUE(hash->insertValue(first_idx, value_to_insert));

    // Insert Multiple
    std::vector<Index3D> indices_to_insert = {Index3D(0, 1, 0), Index3D(0, 0, 1), Index3D(1, 0, 0)};
    std::vector<FloatBlock::Ptr> values_to_insert;
    for (size_t i = 0; i < indices_to_insert.size(); i++) {
        FloatBlock::Ptr v = std::make_shared<FloatBlock>(MemoryType::kHost);
        v->setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.emplace_back(v);
    }
    std::vector<bool> inserted;
    hash->insertValues(indices_to_insert, values_to_insert, inserted);
    for (size_t i = 0; i < indices_to_insert.size(); i++) {
        if (i < 2)
            EXPECT_TRUE(inserted[i]);
        else
            EXPECT_FALSE(inserted[i]);
    }

    // Find All
    std::vector<bool> found;
    std::vector<FloatBlock::Ptr> values_to_find;
    hash->findValues(indices_to_insert, values_to_find, found);
    for (size_t i = 0; i < indices_to_insert.size(); i++) {
        EXPECT_TRUE(found[i]);
        if (i < 2)
            EXPECT_DOUBLE_EQ(values_to_find[i]->getVoxel(Index3D(0, 1, 0)), i + 1);
        else
            EXPECT_DOUBLE_EQ(values_to_find[i]->getVoxel(Index3D(0, 1, 0)), 10.0);
    }
}

TEST(TestCpuHash, InsertManyGetAll) {
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<CPUHashStrategy<FloatBlock>>();

    size_t N = 10;
    std::vector<Index3D> indices_to_insert;
    std::vector<FloatBlock::Ptr> values_to_insert;
    for (size_t i = 0; i < N; i++) {
        indices_to_insert.emplace_back(Index3D(i, 0, 0));
        FloatBlock::Ptr v = std::make_shared<FloatBlock>(MemoryType::kHost);
        v->setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.emplace_back(v);
    }
    std::vector<bool> inserted;
    hash->insertValues(indices_to_insert, values_to_insert, inserted);
    for (size_t i = 0; i < N; i++) EXPECT_TRUE(inserted[i]);

    std::vector<Index3D> all_keys = hash->getAllKeys();
    EXPECT_EQ(all_keys.size(), N);
    for (const auto& key : all_keys) {
        EXPECT_GE(key.x, 0);
        EXPECT_LT(key.x, static_cast<int>(N));
    }

    std::vector<FloatBlock::Ptr> all_values = hash->getAllValues();
    EXPECT_EQ(all_values.size(), N);
    for (const auto& value : all_values) {
        float voxel = value->getVoxel(Index3D(0, 1, 0));

        EXPECT_GE(voxel, 1.0);
        EXPECT_LE(voxel, static_cast<double>(N));
    }

    EXPECT_EQ(hash->size(), N);
}

TEST(TestCpuHash, InsertManyEraseManyGetAll) {
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<CPUHashStrategy<FloatBlock>>();

    size_t N = 10;
    std::vector<Index3D> indices_to_insert;
    std::vector<FloatBlock::Ptr> values_to_insert;
    for (size_t i = 0; i < N; i++) {
        indices_to_insert.emplace_back(Index3D(i, 0, 0));
        FloatBlock::Ptr v = std::make_shared<FloatBlock>(MemoryType::kHost);
        v->setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.emplace_back(v);
    }
    std::vector<bool> inserted;
    hash->insertValues(indices_to_insert, values_to_insert, inserted);
    for (size_t i = 0; i < N; i++) EXPECT_TRUE(inserted[i]);

    std::vector<Index3D> indices_to_erase;
    for (size_t i = 0; i < 2; i++) indices_to_erase.emplace_back(indices_to_insert[i]);

    std::vector<bool> erased;
    hash->eraseValues(indices_to_erase, erased);
    for (size_t i = 0; i < 2; i++) EXPECT_TRUE(erased[i]);

    std::vector<Index3D> all_keys = hash->getAllKeys();
    EXPECT_EQ(all_keys.size(), N - 2);
    for (const auto& key : all_keys) {
        EXPECT_GE(key.x, 2);
        EXPECT_LT(key.x, static_cast<int>(N));
    }

    std::vector<FloatBlock::Ptr> all_values = hash->getAllValues();
    EXPECT_EQ(all_values.size(), N - 2);
    for (const auto& value : all_values) {
        float voxel = value->getVoxel(Index3D(0, 1, 0));

        EXPECT_GE(voxel, 3.0);
        EXPECT_LE(voxel, static_cast<double>(N));
    }

    EXPECT_EQ(hash->size(), N - 2);
}