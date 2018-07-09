/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2017 by Contributors
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu, Da Zheng
*/

#include "./attention_convolution-inl.h"
#include <vector>
// #include "./depthwise_convolution-inl.h"
// #if MXNET_USE_CUDNN == 1
// #include "./cudnn/cudnn_convolution-inl.h"
// #endif  // MXNET_USE_CUDNN


// namespace mshadow {
// namespace cuda {
// template<typename DType>
// __global__ void elemwise_mul_kernel(const int nthreads, const DType* A, const DType* B,
//                                     DType* C) {
//   CUDA_KERNEL_LOOP(index, nthreads) {
//     C[index] = A[index] * B[index];
//   }
// }
// }  // namespace cuda

// // element-wise multiply
// template<typename DType>
// void mxnet_mul(const Tensor<gpu, 2, DType>& A, const Tensor<gpu, 2, DType>& B,
//                const Tensor<gpu, 2, DType>& C) {
//   // TODO: complete this function
//   // get data dptr
//   DType *A_dptr = A.dptr_;
//   DType *B_dptr = B.dptr_;
//   DType *C_dptr = C.dptr_;

//   // get the stream
//   cudaStream_t C_stream = Stream<gpu>::GetStream(C.stream_);

//   // prepare the cuda kernel parameters
//   int nthreads = A.size(0) * A.size(1);
//   const int GridSize = (nthreads + cuda::kMaxThreadsPerBlock - 1) / cuda::kMaxThreadsPerBlock;

//   // launch the cuda kernel
//   cuda::elemwise_mul_kernel<DType><<<GridSize, cuda::kMaxThreadsPerBlock, 0, C_stream>>>(
//     nthreads, A_dptr, B_dptr, C_dptr);
// }
// }  // namespace mshadow


namespace mxnet {
namespace op {

NNVM_REGISTER_OP(AttentionConvolution)
.set_attr<FCompute>("FCompute<gpu>", AttentionConvolutionCompute<gpu>);

NNVM_REGISTER_OP(_backward_AttentionConvolution)
.set_attr<FCompute>("FCompute<gpu>", AttentionConvolutionGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

