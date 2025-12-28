#include <gtest/gtest.h>
#include <voxhash/core/block.h>
#include <voxhash/core/cuda_utils.h>

using namespace voxhash;

TEST(TestBlock, CreateSetAndGet) {
    MemoryType type = MemoryType::kDevice;

    EXPECT_NO_THROW({
        Block<float> b(type);
        Index3D index_to_set(0, 0, 1);
        float test_voxel_value = 100;
        b.setVoxel(index_to_set, test_voxel_value);
        float retrieved_voxel_value = b.getVoxel(index_to_set);
        EXPECT_EQ(test_voxel_value, retrieved_voxel_value);
    });
}

TEST(TestBlock, CopySetAndGet) {
    MemoryType type = MemoryType::kDevice;

    EXPECT_NO_THROW({
        Block<float> b(type);
        Index3D index_to_set(0, 0, 1);
        float test_voxel_value = 100;
        b.setVoxel(index_to_set, test_voxel_value);

        Block<float>::Ptr b_h = Block<float>::copyFrom(b, MemoryType::kHost);
        Index3D index_to_set2(0, 0, 2);
        float test_voxel_value2 = 200;
        b_h->setVoxel(index_to_set2, test_voxel_value2);

        float retrieved_voxel_value = b_h->getVoxel(index_to_set);
        float retrieved_voxel_value2 = b_h->getVoxel(index_to_set2);

        EXPECT_EQ(test_voxel_value, retrieved_voxel_value);
        EXPECT_EQ(test_voxel_value2, retrieved_voxel_value2);
    });
}

TEST(TestBlock, Clear) {
    MemoryType type = MemoryType::kDevice;

    EXPECT_NO_THROW({
        Block<float> b(type);
        Index3D index_to_set(0, 0, 1);
        float test_voxel_value = 100;
        b.setVoxel(index_to_set, test_voxel_value);

        float retrieved_voxel_value = b.getVoxel(index_to_set);
        b.clear();
        float retrieved_voxel_value2 = b.getVoxel(index_to_set);

        EXPECT_EQ(test_voxel_value, retrieved_voxel_value);
        EXPECT_NE(test_voxel_value, retrieved_voxel_value2);
    });
}

TEST(TestBlock, SetFrom) {
    MemoryType type = MemoryType::kDevice;

    EXPECT_NO_THROW({
        Block<float> b(type);
        Index3D index_to_set(0, 0, 1);
        float test_voxel_value = 100;
        b.setVoxel(index_to_set, test_voxel_value);

        Block<float>::Ptr b_h = Block<float>::copyFrom(b, MemoryType::kHost);
        b_h->clear();
        EXPECT_NE(b_h->getVoxel(index_to_set), test_voxel_value);

        EXPECT_TRUE(b_h->setFrom(b));
        EXPECT_EQ(b_h->getVoxel(index_to_set), test_voxel_value);
    });
}

TEST(TestBlock, Release) {
    MemoryType type = MemoryType::kDevice;

    EXPECT_NO_THROW({
        Block<float> b(type);
        Index3D index_to_set(0, 0, 1);
        float test_voxel_value = 100;
        b.setVoxel(index_to_set, test_voxel_value);

        EXPECT_TRUE(b.valid());
        float* ptr = b.release();
        EXPECT_FALSE(b.valid());
        CUDA_CHECK(cudaFree(ptr));
    });
}