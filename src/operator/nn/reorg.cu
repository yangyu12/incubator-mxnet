#include <mshadow/tensor.h>
#include "./reorg-inl.h"
#include "../mxnet_op.h"

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void ReorgForwardKernel(
    int n, const DType* data,
    int data_height, int data_width, int out_height, int out_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int dilate_h, int dilate_w, int pad_h, int pad_w,
    DType* out) {

    // each kernel is for each sample n == C * s * s * H' * W'
    CUDA_KERNEL_LOOP(index, n) {
        int out_w = index % out_width;
        int out_h = index / out_width % out_height;
        int s_w = index / out_width / out_height % kernel_w;
        int s_h = index / out_width / out_height / kernel_w % kernel_h;
        int ch = index / out_width / out_height / kernel_w / kernel_h;

        int data_w = out_w * stride_w - pad_w + s_w * dilate_w;
        int data_h = out_h * stride_h - pad_h + s_h * dilate_h;

        int data_dim = data_height * data_width;
        int out_dim = kernel_h * kernel_w * out_height * out_width;

        out[ch * out_dim + index] =
            (0 <= data_h && data_h < data_height && 0 <= data_w && data_w < data_width) ?
            data[ch * data_dim + data_h * data_height + data_w] : static_cast<DType>(0);
    }
}

template<typename DType>
__global__ void ReorgBackwardKernel(
    int n, const DType* dOut,
    int data_height, int data_width, int out_height, int out_width,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int dilate_h, int dilate_w, int pad_h, int pad_w,
    DType* dData) {

    // each kernel is for each sample n == C * s * s * H' * W'
    CUDA_KERNEL_LOOP(index, n) {
        int out_w = index % out_width;
        int out_h = index / out_width % out_height;
        int s_w = index / out_width / out_height % kernel_w;
        int s_h = index / out_width / out_height / kernel_w % kernel_h;
        int ch = index / out_width / out_height / kernel_w / kernel_h;

        int data_w = out_w * stride_w - pad_w + s_w * dilate_w;
        int data_h = out_h * stride_h - pad_h + s_h * dilate_h;

        int data_dim = data_height * data_width;
        int out_dim = kernel_h * kernel_w * out_height * out_width;

        if (0 <= data_h && data_h < data_height && 0 <= data_w && data_w < data_width)
            dData[ch * data_dim + data_h * data_height + data_w] += dOut[ch * out_dim + index];
    }
}
}  // namespace cuda

template<typename DType>
inline void ReorgForward(
    const Tensor<gpu, 4, DType> &data, const Tensor<gpu, 4, DType> &out,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int dilate_h, int dilate_w, int pad_h, int pad_w) {
    using namespace mxnet::op::mxnet_op;

    // get the shape info
    int num = data.shape_[0];
    int channels = data.shape_[1];
    int data_height = data.shape_[2];
    int data_width = data.shape_[3];

    // calculate the out_height and out_width
    int out_width = ((data_width + 2 * pad_w) - (kernel_w - 1) * dilate_w) / stride_w;
    int out_height = ((data_height + 2 * pad_h) - (kernel_h - 1) * dilate_h) / stride_h;

    int num_threads = channels * kernel_h * kernel_w * out_height * out_width;
    for (int n = 0; n < num; ++n) {
        Tensor<gpu, 3, DType> input_3d = data[n];
        Tensor<gpu, 3, DType> output_3d = out[n];
        cuda::ReorgForwardKernel<DType>
            <<<cuda_get_num_blocks(num_threads), kBaseThreadNum,
               0, Stream<gpu>::GetStream(output_3d.stream_)>>>(
            num_threads, input_3d.dptr<DType>(), data_height, data_width, out_height, out_width,
            kernel_h, kernel_w, stride_h, stride_w, dilate_h, dilate_w, pad_h, pad_w, output_3d.dptr<DType>());
        MSHADOW_CUDA_POST_KERNEL_CHECK(ReorgForwardKernel);
    }
}

template<typename DType>
inline void ReorgBackward(
    const Tensor<gpu, 4, DType> dData, const Tensor<gpu, 4, DType> dOut,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int dilate_h, int dilate_w, int pad_h, int pad_w) {
    using namespace mxnet::op::mxnet_op;

    // get the shape info
    int num = dData.shape_[0];
    int channels = dData.shape_[1];
    int data_height = dData.shape_[2];
    int data_width = dData.shape_[3];

    // calculate the out_height and out_width
    int out_height = dOut.shape_[2];
    int out_width = dOut.shape_[3];

    int num_threads = channels * kernel_h * kernel_w * out_height * out_width;
    for (int n = 0; n < num; ++n) {
        Tensor<gpu, 3, DType> ingrad_3d = dData[n];
        Tensor<gpu, 3, DType> outgrad_3d = dOut[n];
        cuda::ReorgBackwardKernel<DType>
            <<<cuda_get_num_blocks(num_threads), kBaseThreadNum,
               0, Stream<gpu>::GetStream(ingrad_3d.stream_)>>>(
            num_threads, outgrad_3d.dptr<DType>(), data_height, data_width, out_height, out_width,
            kernel_h, kernel_w, stride_h, stride_w, dilate_h, dilate_w, pad_h, pad_w, ingrad_3d.dptr<DType>());
        MSHADOW_CUDA_POST_KERNEL_CHECK(ReorgBackwardKernel);
    }
}

}  // namespace mshadow

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(Reorg)
.set_attr<FCompute>("FCompute<gpu>", ReorgCompute<gpu>);

NNVM_REGISTER_OP(_backward_Reorg)
.set_attr<FCompute>("FCompute<gpu>", ReorgGradCompute<gpu>);

}
}
