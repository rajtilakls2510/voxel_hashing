#include <gtest/gtest.h>
#include <voxhash/core/block.h>
#include <voxhash/core/cuda_utils.h>
#include <voxhash/core/hash.h>

#include <chrono>
#include <memory>

using namespace voxhash;
using FloatBlock = Block<float>;
using namespace std::chrono;

// =========================== Test CPU Hash Strategy ===========================

TEST(TestCpuHash, FindOneInsertManyFindMany) {
    std::shared_ptr<CudaStream> stream = std::make_shared<CudaStreamOwning>();
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<CPUHashStrategy<FloatBlock>>();

    // Search for a non-existing value in the hash
    Vector<Index3D> indices_to_search(1, MemoryType::kHost);
    indices_to_search.data()[0] = Index3D(0, 1, 0);
    auto find_result = hash->findValues(indices_to_search, *stream);
    EXPECT_EQ(find_result.first.data()[0], 0);

    // Insert values
    size_t num_values_to_insert = 1024;
    Vector<Index3D> indices_to_insert(num_values_to_insert, MemoryType::kHost);
    Vector<float*> values_to_insert(num_values_to_insert, MemoryType::kHost);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        indices_to_insert.data()[i] = Index3D(i, 0, 0);
        FloatBlock f(MemoryType::kHost);
        f.setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.data()[i] = f.release();
    }
    auto start_t = high_resolution_clock::now();
    Vector<Bool> inserted = hash->insertValues(indices_to_insert, values_to_insert, *stream);
    auto end_t = high_resolution_clock::now();
    std::cout << "Time taken to insert " << num_values_to_insert
              << " values: " << duration_cast<microseconds>(end_t - start_t).count() << " us\n";
    for (size_t i = 0; i < num_values_to_insert; i++) {
        EXPECT_EQ(inserted.data()[i], 1);
    }

    // Find inserted values
    size_t num_indices_to_search = 1026;
    Vector<Index3D> indices_to_search2(num_indices_to_search, MemoryType::kHost);
    for (size_t i = 0; i < num_indices_to_search; i++) {
        indices_to_search2.data()[i] = Index3D(i, 0, 0);
    }
    auto find_result2 = hash->findValues(indices_to_search2, *stream);
    for (size_t i = 0; i < num_indices_to_search; i++) {
        if (i < num_values_to_insert) {
            EXPECT_EQ(find_result2.first.data()[i], 1);
            EXPECT_DOUBLE_EQ(
                    *(find_result2.second.data()[i] + FloatBlock::idx(Index3D(0, 1, 0))), i + 1);
        } else {
            EXPECT_EQ(find_result2.first.data()[i], 0);
        }
    }
}

TEST(TestCpuHash, InsertManyGetAll) {
    std::shared_ptr<CudaStream> stream = std::make_shared<CudaStreamOwning>();
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<CPUHashStrategy<FloatBlock>>();

    // Insert values
    size_t num_values_to_insert = 512;
    Vector<Index3D> indices_to_insert(num_values_to_insert, MemoryType::kHost);
    Vector<float*> values_to_insert(num_values_to_insert, MemoryType::kHost);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        indices_to_insert.data()[i] = Index3D(i, 0, 0);
        FloatBlock f(MemoryType::kHost);
        f.setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.data()[i] = f.release();
    }
    Vector<Bool> inserted = hash->insertValues(indices_to_insert, values_to_insert, *stream);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        EXPECT_EQ(inserted.data()[i], 1);
    }

    // Insert another set of values
    Vector<Index3D> indices_to_insert2(num_values_to_insert, MemoryType::kHost);
    Vector<float*> values_to_insert2(num_values_to_insert, MemoryType::kHost);
    for (size_t i = num_values_to_insert; i < 2 * num_values_to_insert; i++) {
        indices_to_insert2.data()[i - num_values_to_insert] = Index3D(i, 0, 0);
        FloatBlock f(MemoryType::kHost);
        f.setVoxel(Index3D(0, 1, 0), i - num_values_to_insert + 1);
        values_to_insert2.data()[i - num_values_to_insert] = f.release();
    }
    Vector<Bool> inserted2 = hash->insertValues(indices_to_insert2, values_to_insert2, *stream);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        EXPECT_EQ(inserted2.data()[i], 1);
    }

    auto result = hash->getAllKeyValues(*stream);
    EXPECT_EQ(result.first.size(), hash->size());
    EXPECT_EQ(result.second.size(), hash->size());

    for (size_t i = 0; i < result.first.size(); i++) {
        EXPECT_GE(result.first.data()[i].x, 0);
        EXPECT_LE(result.first.data()[i].x, 2 * num_values_to_insert);
    }

    for (size_t i = 0; i < result.second.size(); i++) {
        EXPECT_GE(*(result.second.data()[i] + FloatBlock::idx(Index3D(0, 1, 0))), 1);
        EXPECT_LE(
                *(result.second.data()[i] + FloatBlock::idx(Index3D(0, 1, 0))),
                2 * num_values_to_insert + 1);
    }
}

TEST(TestCpuHash, InsertManyEraseManyFindMany) {
    std::shared_ptr<CudaStream> stream = std::make_shared<CudaStreamOwning>();
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<CPUHashStrategy<FloatBlock>>();

    // Insert values
    size_t num_values_to_insert = 1024;
    Vector<Index3D> indices_to_insert(num_values_to_insert, MemoryType::kHost);
    Vector<float*> values_to_insert(num_values_to_insert, MemoryType::kHost);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        indices_to_insert.data()[i] = Index3D(i, 0, 0);
        FloatBlock f(MemoryType::kHost);
        f.setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.data()[i] = f.release();
    }
    Vector<Bool> inserted = hash->insertValues(indices_to_insert, values_to_insert, *stream);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        EXPECT_EQ(inserted.data()[i], 1);
    }

    // Erase the first half of values
    size_t num_values_to_erase = num_values_to_insert / 2;
    Vector<Index3D> indices_to_erase(num_values_to_erase, MemoryType::kHost);
    for (size_t i = 0; i < num_values_to_erase; i++) {
        indices_to_erase.data()[i] = Index3D(i, 0, 0);
    }
    Vector<Bool> erased = hash->eraseValues(indices_to_erase, *stream);
    for (size_t i = 0; i < num_values_to_erase; i++) {
        EXPECT_EQ(erased.data()[i], 1);
    }

    // Find all keys and check if they exist in hash
    auto result = hash->findValues(indices_to_insert, *stream);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        if (i < num_values_to_erase)
            EXPECT_EQ(result.first.data()[i], 0);
        else {
            EXPECT_EQ(result.first.data()[i], 1);
            EXPECT_DOUBLE_EQ(*(result.second.data()[i] + FloatBlock::idx(Index3D(0, 1, 0))), i + 1);
        }
    }
}

// =========================== Test GPU Hash Strategy ===========================

TEST(TestGpuHash, FindOneInsertManyFindMany) {
    std::shared_ptr<CudaStream> stream = std::make_shared<CudaStreamOwning>();
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<GPUHashStrategy<FloatBlock>>(5);

    // Search for a non-existing value in the hash
    Vector<Index3D> indices_to_search(1, MemoryType::kHost);
    indices_to_search.data()[0] = Index3D(0, 1, 0);
    auto find_result = hash->findValues(indices_to_search, *stream);
    Vector<Bool>::Ptr found_h =
            Vector<Bool>::copyFrom(find_result.first, MemoryType::kHost, *stream);
    stream->synchronize();
    EXPECT_EQ(found_h->data()[0], 0);

    // Insert values
    size_t num_values_to_insert = 1024;
    Vector<Index3D> indices_to_insert(num_values_to_insert, MemoryType::kHost);
    Vector<float*> values_to_insert(num_values_to_insert, MemoryType::kHost);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        indices_to_insert.data()[i] = Index3D(i, 0, 0);
        FloatBlock f(MemoryType::kHost);
        f.setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.data()[i] = f.release();
    }
    auto start_t = high_resolution_clock::now();
    Vector<Bool> inserted = hash->insertValues(indices_to_insert, values_to_insert, *stream);
    auto end_t = high_resolution_clock::now();
    std::cout << "Time taken to insert " << num_values_to_insert
              << " values: " << duration_cast<microseconds>(end_t - start_t).count() << " us\n";
    Vector<Bool>::Ptr inserted_h = Vector<Bool>::copyFrom(inserted, MemoryType::kHost, *stream);
    stream->synchronize();
    for (size_t i = 0; i < num_values_to_insert; i++) {
        EXPECT_EQ(inserted_h->data()[i], 1);
    }

    // Find inserted values
    size_t num_indices_to_search = 1026;
    Vector<Index3D> indices_to_search2(num_indices_to_search, MemoryType::kHost);
    for (size_t i = 0; i < num_indices_to_search; i++) {
        indices_to_search2.data()[i] = Index3D(i, 0, 0);
    }
    auto find_result2 = hash->findValues(indices_to_search2, *stream);

    Vector<Bool>::Ptr found_h2 =
            Vector<Bool>::copyFrom(find_result2.first, MemoryType::kHost, *stream);
    Vector<float*>::Ptr values_h2 =
            Vector<float*>::copyFrom(find_result2.second, MemoryType::kHost, *stream);
    stream->synchronize();
    for (size_t i = 0; i < num_indices_to_search; i++) {
        if (i < num_values_to_insert) {
            EXPECT_EQ(found_h2->data()[i], 1);
            EXPECT_DOUBLE_EQ(*(values_h2->data()[i] + FloatBlock::idx(Index3D(0, 1, 0))), i + 1);
        } else {
            EXPECT_EQ(found_h2->data()[i], 0);
        }
    }
}

TEST(TestGpuHash, InsertManyGetAll) {
    std::shared_ptr<CudaStream> stream = std::make_shared<CudaStreamOwning>();
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<GPUHashStrategy<FloatBlock>>(5);

    // Insert values
    size_t num_values_to_insert = 512;
    Vector<Index3D> indices_to_insert(num_values_to_insert, MemoryType::kHost);
    Vector<float*> values_to_insert(num_values_to_insert, MemoryType::kHost);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        indices_to_insert.data()[i] = Index3D(i, 0, 0);
        FloatBlock f(MemoryType::kHost);
        f.setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.data()[i] = f.release();
    }
    Vector<Bool> inserted = hash->insertValues(indices_to_insert, values_to_insert, *stream);
    Vector<Bool>::Ptr inserted_h = Vector<Bool>::copyFrom(inserted, MemoryType::kHost, *stream);
    stream->synchronize();
    for (size_t i = 0; i < num_values_to_insert; i++) {
        EXPECT_EQ(inserted_h->data()[i], 1);
    }

    // Insert another set of values
    Vector<Index3D> indices_to_insert2(num_values_to_insert, MemoryType::kHost);
    Vector<float*> values_to_insert2(num_values_to_insert, MemoryType::kHost);
    for (size_t i = num_values_to_insert; i < 2 * num_values_to_insert; i++) {
        indices_to_insert2.data()[i - num_values_to_insert] = Index3D(i, 0, 0);
        FloatBlock f(MemoryType::kHost);
        f.setVoxel(Index3D(0, 1, 0), i - num_values_to_insert + 1);
        values_to_insert2.data()[i - num_values_to_insert] = f.release();
    }
    Vector<Bool> inserted2 = hash->insertValues(indices_to_insert2, values_to_insert2, *stream);
    Vector<Bool>::Ptr inserted_h2 = Vector<Bool>::copyFrom(inserted2, MemoryType::kHost, *stream);
    stream->synchronize();
    for (size_t i = 0; i < num_values_to_insert; i++) {
        EXPECT_EQ(inserted_h2->data()[i], 1);
    }

    auto result = hash->getAllKeyValues(*stream);
    Vector<Index3D>::Ptr all_keys =
            Vector<Index3D>::copyFrom(result.first, MemoryType::kHost, *stream);
    Vector<float*>::Ptr all_values =
            Vector<float*>::copyFrom(result.second, MemoryType::kHost, *stream);
    stream->synchronize();
    EXPECT_EQ(all_keys->size(), hash->size());
    EXPECT_EQ(all_values->size(), hash->size());

    for (size_t i = 0; i < all_keys->size(); i++) {
        EXPECT_GE(all_keys->data()[i].x, 0);
        EXPECT_LE(all_keys->data()[i].x, 2 * num_values_to_insert);
    }

    for (size_t i = 0; i < all_values->size(); i++) {
        EXPECT_GE(*(all_values->data()[i] + FloatBlock::idx(Index3D(0, 1, 0))), 1);
        EXPECT_LE(
                *(all_values->data()[i] + FloatBlock::idx(Index3D(0, 1, 0))),
                2 * num_values_to_insert + 1);
    }
}

TEST(TestGpuHash, InsertManyEraseManyFindMany) {
    std::shared_ptr<CudaStream> stream = std::make_shared<CudaStreamOwning>();
    std::shared_ptr<HashStrategy<FloatBlock>> hash =
            std::make_shared<GPUHashStrategy<FloatBlock>>(5);

    // Insert values
    size_t num_values_to_insert = 1024;
    Vector<Index3D> indices_to_insert(num_values_to_insert, MemoryType::kHost);
    Vector<float*> values_to_insert(num_values_to_insert, MemoryType::kHost);
    for (size_t i = 0; i < num_values_to_insert; i++) {
        indices_to_insert.data()[i] = Index3D(i, 0, 0);
        FloatBlock f(MemoryType::kHost);
        f.setVoxel(Index3D(0, 1, 0), i + 1);
        values_to_insert.data()[i] = f.release();
    }
    Vector<Bool> inserted = hash->insertValues(indices_to_insert, values_to_insert, *stream);
    Vector<Bool>::Ptr inserted_h = Vector<Bool>::copyFrom(inserted, MemoryType::kHost, *stream);
    stream->synchronize();
    for (size_t i = 0; i < num_values_to_insert; i++) {
        EXPECT_EQ(inserted_h->data()[i], 1);
    }

    // Erase the first half of values
    size_t num_values_to_erase = num_values_to_insert / 2;
    Vector<Index3D> indices_to_erase(num_values_to_erase, MemoryType::kHost);
    for (size_t i = 0; i < num_values_to_erase; i++) {
        indices_to_erase.data()[i] = Index3D(i, 0, 0);
    }
    Vector<Bool> erased = hash->eraseValues(indices_to_erase, *stream);
    Vector<Bool>::Ptr erased_h = Vector<Bool>::copyFrom(erased, MemoryType::kHost, *stream);
    stream->synchronize();
    for (size_t i = 0; i < num_values_to_erase; i++) {
        EXPECT_EQ(erased_h->data()[i], 1);
    }

    // Find all keys and check if they exist in hash
    auto result = hash->findValues(indices_to_insert, *stream);
    Vector<Bool>::Ptr found_h = Vector<Bool>::copyFrom(result.first, MemoryType::kHost, *stream);
    Vector<float*>::Ptr values_h =
            Vector<float*>::copyFrom(result.second, MemoryType::kHost, *stream);
    stream->synchronize();
    for (size_t i = 0; i < num_values_to_insert; i++) {
        if (i < num_values_to_erase)
            EXPECT_EQ(found_h->data()[i], 0);
        else {
            EXPECT_EQ(found_h->data()[i], 1);
            EXPECT_DOUBLE_EQ(*(values_h->data()[i] + FloatBlock::idx(Index3D(0, 1, 0))), i + 1);
        }
    }
}