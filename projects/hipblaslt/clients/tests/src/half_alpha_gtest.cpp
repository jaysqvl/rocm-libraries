// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

namespace
{
    class HalfAlpha : public ::testing::Test
    {
    protected:
        hipblasLtHandle_t       handle = nullptr;
        hipblasLtMatmulDesc_t   desc   = nullptr;
        hipblasLtMatrixLayout_t layout = nullptr;

        void SetUp() override
        {
            ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_16F, HIP_R_16F),
                      HIPBLAS_STATUS_SUCCESS);
            ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout, HIP_R_16F, 1, 1, 1),
                      HIPBLAS_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(layout)
                hipblasLtMatrixLayoutDestroy(layout);
            if(desc)
                hipblasLtMatmulDescDestroy(desc);
            if(handle)
                hipblasLtDestroy(handle);
        }
    };

    // Keep alpha outside computeTypeInterface: its 16-byte storage masks a
    // float-sized read of a half scalar (ROCM-31553). A null A forces an
    // argument error before any matrix pointer is used to launch a kernel.
    TEST_F(HalfAlpha, quick_MatmulRejectsNullAWithStandaloneHalfAlpha)
    {
        hipblasLtHalf alpha = 1.0f;
        hipblasLtHalf beta  = 0.0f;
        hipblasLtHalf data  = 0.0f;
        static_assert(sizeof(alpha) == 2);

        EXPECT_EQ(hipblasLtMatmul(handle,
                                 desc,
                                 &alpha,
                                 nullptr,
                                 layout,
                                 &data,
                                 layout,
                                 &beta,
                                 &data,
                                 layout,
                                 &data,
                                 layout,
                                 nullptr,
                                 nullptr,
                                 0,
                                 nullptr),
                  HIPBLAS_STATUS_INVALID_VALUE);
    }

    TEST_F(HalfAlpha, quick_SetProblemRejectsNullAWithStandaloneHalfAlpha)
    {
        hipblasLtHalf alpha = 1.0f;
        hipblasLtHalf beta  = 0.0f;
        hipblasLtHalf data  = 0.0f;
        static_assert(sizeof(alpha) == 2);

        hipblaslt_ext::Gemm gemm(handle,
                                HIPBLAS_OP_N,
                                HIPBLAS_OP_N,
                                HIP_R_16F,
                                HIP_R_16F,
                                HIP_R_16F,
                                HIP_R_16F,
                                HIPBLAS_COMPUTE_16F);
        hipblaslt_ext::GemmEpilogue epilogue;
        hipblaslt_ext::GemmInputs   inputs;
        inputs.setA(nullptr);
        inputs.setB(&data);
        inputs.setC(&data);
        inputs.setD(&data);
        inputs.setAlpha(&alpha);
        inputs.setBeta(&beta);

        EXPECT_EQ(gemm.setProblem(1, 1, 1, 1, epilogue, inputs), HIPBLAS_STATUS_INVALID_VALUE);
    }
}
