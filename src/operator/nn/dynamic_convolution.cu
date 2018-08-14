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

#include "./dynamic_convolution-inl.h"
#include <vector>
// #include "./depthwise_convolution-inl.h"
#include "../mxnet_op.h"

namespace mshadow {
using namespace mxnet;
using namespace mxnet::op;

template<typename DType>
__global__ void dynconv_im2col_gpu_kernel(
  const int n, const DType* data_im,
  const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int dilation_h, const int dilation_w,
  const int h_samples, const int w_samples, const int kernel_stride, DType* data_col) {
  //
  CUDA_KERNEL_LOOP(index, n) {
    // data_col shape (H'*W'*s*s, Cin*k*k)
    // index for data_col
    const int c_ind = index % channels;
    const int w_s_ind = index / channels % w_samples;
    const int h_s_ind = index / channels / w_samples % h_samples;
    const int w_ind = index / channels / w_samples / h_samples % width;
    const int h_ind = index / channels / w_samples / h_samples / width % height;
    const int n_ind = index / channels / w_samples / h_samples / width / height;  // it should always be 0
    const int w_sample_pad = (w_samples - 1) / 2;
    const int h_sample_pad = (h_samples - 1) / 2;

    // index for data_im
    const int b_c_ind = c_ind;
    const int b_w_ind = w_ind + (w_s_ind - w_samples + w_sample_pad + 1) * kernel_stride;
    const int b_h_ind = h_ind + (h_s_ind - h_samples + h_sample_pad + 1) * kernel_stride;

    //
    const int top_offset = index * kernel_h * kernel_w;
    const int bottom_offset = n_ind * channels * width * height + b_c_ind * width * height +  b_h_ind * width + b_w_ind;
    const int w_pad = (kernel_w - 1) / 2;
    const int h_pad = (kernel_h - 1) / 2;

    // im2col loop
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        data_col[top_offset + i * kernel_w + j] = 0;
        if (b_h_ind + dilation_h * (i - kernel_h + 1 + h_pad) < height &&
            b_h_ind + dilation_h * (i - kernel_h + h_pad + 1) >= 0) {
          if (b_w_ind + dilation_w * (j - kernel_w + w_pad + 1) < width &&
              b_w_ind + dilation_w * (j - kernel_w + w_pad + 1) >=0) {
              data_col[top_offset + i * kernel_w + j] =
              data_im[bottom_offset + dilation_h * (i - kernel_h + 1 + h_pad ) * width + dilation_w * (j - kernel_w + 1 + w_pad)];
          }
        }
      }
    }
  }
}

template<typename DType>
inline void dynconv_im2col(
  Stream<gpu>* s,
  const DType* data_im, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int dilation_h, const int dilation_w,
  const int h_samples, const int w_samples, const int kernel_stride, DType* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  using namespace mxnet_op;
  int pad_h = (kernel_h - 1)/2;
  int pad_w = (kernel_w - 1)/2;
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) + 1;
  int num_threads = channels * height_col * width_col * h_samples * w_samples;
  // NOLINT_NEXT_LINE(whitespace/operators)
  dynconv_im2col_gpu_kernel<DType>
      <<<cuda_get_num_blocks(num_threads), cuda::kBaseThreadNum,
         0, Stream<gpu>::GetStream(s)>>>(
      num_threads, data_im, channels, height, width, kernel_h, kernel_w, dilation_h, dilation_w,
      h_samples, w_samples, kernel_stride, data_col);
  MSHADOW_CUDA_POST_KERNEL_CHECK(dynconv_im2col_gpu_kernel);
}

/*!
 *
 */
template <typename DType>
__global__ void dynconv_col2im_gpu_kernel(
  const int n, const DType* data_col, //from HWssCkk to CHW, n is CHW
  const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int dilation_h, const int dilation_w,
  const int h_samples, const int w_samples, const int kernel_stride,
  DType* data_im) {
  //
  CUDA_KERNEL_LOOP(index, n) {  // n should be C H W
    DType val = 0;
    const int w_im = index % width ;
    const int h_im = (index / width) % height ;
    const int c_im = index / (width * height);
    const int h_sample_pad = (h_samples + 1)/2;
    const int w_sample_pad = (w_samples + 1)/2;
    const int h_pad = (kernel_h + 1)/2;
    const int w_pad = (kernel_w + 1)/2;
    for (int s_h = 0; s_h < h_samples; s_h ++){
      for (int s_w = 0; s_w < w_samples; s_w ++){
        for (int k_h = 0; k_h < kernel_h; k_h ++){
          for (int k_w = 0; k_w < kernel_w; k_w ++){
            int  val_ind_h = h_im + (s_h - h_samples + h_sample_pad) * kernel_stride + (k_h - kernel_h + h_pad)* dilation_h;
            int val_ind_w = w_im + (s_w - w_samples + w_sample_pad) * kernel_stride + (k_w - kernel_w + w_pad)* dilation_w;
            if (val_ind_w>=0 && val_ind_w < width && val_ind_h>=0 && val_ind_h < height){
              const int col_index = ((((h_im*width + w_im)*h_samples + s_h )*w_samples + s_w )*channels + c_im)* kernel_h*kernel_w + k_h * kernel_w  + k_w;
              val += data_col[col_index];
            }
          }
        }
      }
    }
    data_im[index] = val;
  }
}

/*!
 *
 */
template <typename DType>
inline void dynconv_col2im(
  Stream<gpu>* s, const DType* data_col, //from HWssCkk to CHW, n is CHW
  const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int dilation_h, const int dilation_w,
  const int h_samples, const int w_samples, const int kernel_stride,
  DType* data_im) {
  //
  using namespace mxnet_op;
  int num_threads = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  dynconv_col2im_gpu_kernel<DType>
      <<<cuda_get_num_blocks(num_threads), cuda::kBaseThreadNum,
         0, Stream<gpu>::GetStream(s)>>>(
      num_threads, data_col, channels, height, width, kernel_h, kernel_w, dilation_h, dilation_w,
      h_samples, w_samples, kernel_stride, data_im);
  MSHADOW_CUDA_POST_KERNEL_CHECK(dynconv_col2im_gpu_kernel);
}

/*!
 *
 */
template <typename DType>
__global__ void dynconv_inprod_gpu_kernel(
  const int n, const DType* offset_col_buffer, const DType* across_weight, const DType* within_weight,
  const int num_output, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w,
  const int h_samples, const int w_samples,
  DType* top_data) {
  // n == H' * W' * s * s * Cout
  CUDA_KERNEL_LOOP(index, n) {
    // index for output data
    const int output_c = index % num_output;
    const int w_s_ind = index / num_output % w_samples;
    const int h_s_ind = index / num_output / w_samples % h_samples;
    const int w_ind = index / num_output / w_samples / h_samples % width;
    const int h_ind = index / num_output / w_samples / h_samples / width % height;
    const int n_ind = index / num_output / w_samples / h_samples / width / height;
    // printf("n_ind: %d\n", n_ind);

    // H * W * h_samples_ * w_samples_ *C*k*k
    const int offset_col_buffer_offset = (int)(index / num_output) * channels * kernel_h * kernel_w;
    const int across_weight_offset = (n_ind * width * height + h_ind * width + w_ind) * num_output * channels;
    const int within_weight_offset = (n_ind * width * height + h_ind * width + w_ind) * num_output * kernel_h * kernel_w;
    const int top_data_offset = (((n_ind * width * height + h_ind * width + w_ind) * h_samples + h_s_ind ) * w_samples + w_s_ind) * num_output;
    // int nthreads =num_output;
    DType * c = top_data + top_data_offset;
    DType coeff = 0.0;
    const DType* a_across = across_weight + across_weight_offset;
    const DType* a_within = within_weight + within_weight_offset;
    const DType* b = offset_col_buffer + offset_col_buffer_offset;
    top_data[index] *= coeff;
    for (int i = 0; i < channels * kernel_h * kernel_w; i++) {
      const int k_w = i % kernel_w;
      const int k_h = i / kernel_w % kernel_h;
      const int k_c = i / kernel_w / kernel_h;
      const DType w_across_val = (k_w == int((kernel_w - 1) / 2) && k_h == int((kernel_h - 1) / 2)) ? a_across[output_c * channels + k_c]: DType(0.0);
      const DType w_within_val = a_within[output_c * kernel_h * kernel_w + k_h * kernel_w + k_w];
      top_data[index] += ((w_across_val + w_within_val) * b[i]);
      // cd if (index == n -5)printf("hello world\n");
      // printf("index: %d\n",index);
    }
  }
}

/*!
 *
 */
template <typename DType>
inline void dynconv_inprod(
  Stream<gpu>* s,
  const DType* col_buffer, const DType* across_weight, const DType* within_weight,
  const int num_filters, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int h_samples, const int w_samples,
  DType* out_data) {
  //
  using namespace mxnet_op;
  int num_threads = num_filters * height * width * h_samples * w_samples;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  dynconv_inprod_gpu_kernel<DType>
      <<<cuda_get_num_blocks(num_threads), cuda::kBaseThreadNum,
         0, Stream<gpu>::GetStream(s)>>>(
      num_threads, col_buffer, across_weight, within_weight, num_filters, channels, height, width,
      kernel_h, kernel_w, h_samples, w_samples, out_data);
  MSHADOW_CUDA_POST_KERNEL_CHECK(dynconv_inprod_gpu_kernel);
}

/*!
 *
 */
template <typename DType>
__global__ void dynconv_dAWeight_gpu_kernel(
  const int n, const DType* offset_col_buffer, const DType* top_diff,
  const int num_output, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int h_samples, const int w_samples,
  DType* across_weight_diff) {
  // n == H' * W'. Why not H' * W' * Cout * Cin
  CUDA_KERNEL_LOOP(index, n) {
    const int c = index % channels;
    const int output_c = index / channels % num_output;
    const int w_ind = index / channels / num_output % width;
    const int h_ind = index / channels / num_output / width % height;
    // const int kernel_dim_ = channels * kernel_h * kernel_w;

    for (int s_h = 0; s_h < h_samples; s_h++) {
      for (int s_w = 0; s_w < w_samples; s_w++) {
        int buffer_offset = (((h_ind * width + w_ind) * h_samples + s_h) * w_samples + s_w) * channels * kernel_h * kernel_w;
        int top_offset = (((h_ind * width + w_ind) * h_samples + s_h) * w_samples + s_w) * num_output;
        //int weight_offset = index*num_output*channels*kernel_h*kernel_w;
        const int across_weight_offset = (h_ind * width + w_ind) * num_output * channels;
        // const int within_weight_offset = (h_ind * width + w_ind) * num_output * kernel_h * kernel_w;
        //gpu_matrix_mult<DType>( num_output*kernel_dim_, top_diff + top_offset ,offset_col_buffer + buffer_offset,
        //weight_diff + weight_offset, num_output /* C' */, 1,kernel_dim_ /* C * h * w */ ,(DType)1.);

        DType* c_across = across_weight_diff + across_weight_offset;

        DType coeff = 1.0;
        const DType* a = top_diff + top_offset;
        const DType* b = offset_col_buffer + buffer_offset;
        c_across[output_c*channels+c] +=  a[output_c] * b[c*kernel_h*kernel_w + ((kernel_h -1)/2)*kernel_w + (kernel_w-1)/2];
      }
    }
  }
}

/*!
 *
 */
template <typename DType>
inline void dynconv_dAWeight(
  Stream<gpu>* s,
  const DType* col_buffer, const DType* out_grad,
  const int num_filters, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int h_samples, const int w_samples,
  DType* aweight_grad) {
  // H' * W' * Cout * Cin
  using namespace mxnet_op;
  int num_threads = height * width * num_filters * channels;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  dynconv_dAWeight_gpu_kernel<DType>
      <<<cuda_get_num_blocks(num_threads), cuda::kBaseThreadNum,
         0, Stream<gpu>::GetStream(s)>>>(
      num_threads, col_buffer, out_grad, channels,num_filters, height, width,
      kernel_h, kernel_w, h_samples, w_samples, aweight_grad);
  MSHADOW_CUDA_POST_KERNEL_CHECK(dynconv_dAWeight_gpu_kernel);
}

/*!
 *
 */
template <typename DType>
__global__ void dynconv_dWWeight_gpu_kernel(
  const int n, const DType* offset_col_buffer, const DType* top_diff,
  const int num_output, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int h_samples, const int w_samples,
  DType* within_weight_diff) {
  // n == H' W' Cout k*k
  CUDA_KERNEL_LOOP(index, n) {
    const int k_w = index % kernel_w;
    const int k_h = index / kernel_w % kernel_h;
    const int output_c = index / kernel_w / kernel_h % num_output;
    const int w_ind = index / kernel_w / kernel_h / num_output % width;
    const int h_ind = index / kernel_w / kernel_h / num_output / width % height;
    // const int kernel_dim_ = channels * kernel_h * kernel_w;

    for (int s_h = 0; s_h < h_samples; s_h++) {
      for (int s_w = 0; s_w < w_samples; s_w++) {
        int buffer_offset = (((h_ind * width + w_ind) * h_samples + s_h) * w_samples + s_w) * channels * kernel_h * kernel_w;
        int top_offset =  (((h_ind * width + w_ind) * h_samples + s_h) * w_samples + s_w) * num_output;

        const int within_weight_offset = (h_ind * width + w_ind) * num_output * kernel_h * kernel_w;

        DType * c_within = within_weight_diff + within_weight_offset;
        DType coeff = 1.0;
        const DType* a = top_diff + top_offset;
        const DType* b = offset_col_buffer + buffer_offset;
        for (int ind = 0; ind < channels; ind++) {
          c_within[output_c*kernel_h*kernel_w + k_h*kernel_w + k_w] += a[output_c] * b[ind*kernel_h*kernel_w + k_h*kernel_w + k_w];
        }
      }
    }
  }
}

/*!
 *
 */
template <typename DType>
inline void dynconv_dWWeight(Stream<gpu>* s,
  const DType* col_buffer, const DType* out_grad,
  const int num_filters, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int h_samples, const int w_samples,
  DType* wweight_grad) {
  using namespace mxnet_op;
  // H' * W' * Cout * k * k
  int num_threads = num_filters * height * width * kernel_h * kernel_w;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  dynconv_dWWeight_gpu_kernel<DType>
      <<<cuda_get_num_blocks(num_threads), cuda::kBaseThreadNum,
         0, Stream<gpu>::GetStream(s)>>>(
      num_threads, col_buffer, out_grad, num_filters, channels, height, width,
      kernel_h, kernel_w, h_samples, w_samples, wweight_grad);
  MSHADOW_CUDA_POST_KERNEL_CHECK(dynconv_dWWeight_gpu_kernel);
}

/*!
 *
 */
template <typename DType> // n is H W ssCkk , weight is like  W , H, C' C, k, k, (after permuted!!), top_data N, W , H , h_samples_ * w_samples_*C'
__global__ void dynconv_dData_gpu_kernel(
  const int n, const DType* across_weight, const DType* within_weight, const DType* top_diff,
  const int num_output, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int h_samples, const int w_samples,
  DType* offset_col_buffer) {
  // n == H' * W'  why not H' * W' * s * s * Cin * k * k
  CUDA_KERNEL_LOOP(index, n) {
    const int k_w = index % kernel_w;
    const int k_h = index / kernel_w % kernel_h;
    const int k_c = index / kernel_w / kernel_h % channels;
    const int s_w = index / kernel_w / kernel_h / channels % w_samples;
    const int s_h = index / kernel_w / kernel_h / channels / w_samples % h_samples;
    const int w_ind = index / kernel_w / kernel_h / channels / w_samples / h_samples % width;
    const int h_ind = index / kernel_w / kernel_h / channels / w_samples / h_samples / width % height;
    const int kernel_dim_ = channels * kernel_h * kernel_w;
    int buffer_offset = index;
    int top_offset = (int)(index / kernel_dim_) * num_output;
    const int across_weight_offset = (h_ind * width + w_ind) * num_output * channels;
    const int within_weight_offset = (h_ind * width + w_ind) * num_output * kernel_h * kernel_w;
    const DType* a = top_diff + top_offset;
    const DType* b_across = across_weight + across_weight_offset;
    const DType* b_within = within_weight + within_weight_offset;

    offset_col_buffer[index] = 0;
    for (int ind = 0; ind < num_output; ind++) {
      const DType w_across_val = (k_w == int((kernel_w-1)/2) && k_h == int((kernel_h-1)/2)) ? b_across[ind*channels + k_c]: DType(0.0);
      const DType w_within_val = b_within[ind * kernel_h * kernel_w + k_h * kernel_w + k_w];
      offset_col_buffer[index] += a[ind] * (w_across_val + w_within_val);
    }
  }
}

/*!
 *
 */
template <typename DType>
inline void dynconv_dCol(
  Stream<gpu>* s,
  const DType* across_weight, const DType* within_weight, const DType* out_grad,
  const int num_filters, const int channels, const int height, const int width,
  const int kernel_h, const int kernel_w, const int h_samples, const int w_samples,
  DType* col_buffer) {
  using namespace mxnet_op;
  // H' * W' * s * s * Cin * k * k
  // index_t num_threads = out_shape.ProdShape(2, out_shape.ndim());
  int num_threads = height * width * h_samples * w_samples * channels * kernel_h * kernel_w;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  dynconv_dData_gpu_kernel<DType>
      <<<cuda_get_num_blocks(num_threads), cuda::kBaseThreadNum,
         0, Stream<gpu>::GetStream(s)>>>(
      num_threads, across_weight, within_weight, out_grad, num_filters, channels, height, width,
      kernel_h, kernel_w, h_samples, w_samples, col_buffer);
  MSHADOW_CUDA_POST_KERNEL_CHECK(dynconv_dData_gpu_kernel);
}
} // namespace mshadow


namespace mxnet {
namespace op {

NNVM_REGISTER_OP(DynamicConvolution)
.set_attr<FCompute>("FCompute<gpu>", DynamicConvolutionCompute<gpu>);

NNVM_REGISTER_OP(_backward_DynamicConvolution)
.set_attr<FCompute>("FCompute<gpu>", DynamicConvolutionGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

