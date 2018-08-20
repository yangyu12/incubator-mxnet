#include <mshadow/tensor.h>
#include "./radiate_sample-inl.h"
#include "../mxnet_op.h"

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

// TODO: implement the gpu version functions if needed.
namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void RadSamForwardKernel(
	int n, const DType* data, 
	const int data_height, const int data_width, const int out_height, const int out_width,
	const int pad_h, const int pad_w, const int num_group, const int group_size, 
	DType* out) {
	// balabala
	CUDA_KERNEL_LOOP(index, n) {
		int out_w = index % out_width;
		int out_h = index / out_width % out_height;
		int out_ch = index / out_width / out_height;

		int data_w = out_w + (num_group - 1) - pad_w;  // coordinate transform
		int data_h = out_h + (num_group - 1) - pad_h;
		int data_ch = out_ch;

		int group_idx = out_ch / group_size;
		int sample_size = 2 * group_idx + 1;

		if (sample_size == 1) {
			out[index] = (0 <= data_h && data_h < data_height && 0 <= data_w && data_w < data_width) ?
						 data[(data_ch * data_height + data_h) * data_width + data_w] : static_cast<DType>(0);
		} else {
			int sample_lu_w = data_w - group_idx;
			int sample_lu_h = data_h - group_idx;
			int sample_rd_w = data_w + group_idx;
			int sample_rd_h = data_h + group_idx;
			DType val = 0;
			for (int i = 0; i < sample_size - 1; ++i) {
				DType val_to_sum = (0 <= sample_lu_h && sample_lu_h < data_height && 
									0 <= sample_lu_w + i && sample_lu_w + i < data_width) ?
								   data[(data_ch * data_height + sample_lu_h) * data_width + sample_lu_w + i] : static_cast<DType>(0);
				val += val_to_sum;
				val_to_sum = (0 <= sample_lu_h + i && sample_lu_h + i < data_height && 
										     0 <= sample_rd_w && sample_rd_w < data_width) ?
						     data[(data_ch * data_height + sample_lu_h + i) * data_width + sample_rd_w] : static_cast<DType>(0);
				val += val_to_sum;
				val_to_sum = (0 <= sample_rd_h && sample_rd_h < data_height && 
										     0 <= sample_rd_w - i && sample_rd_w - i < data_width) ?
						     data[(data_ch * data_height + sample_rd_h) * data_width + sample_rd_w - i] : static_cast<DType>(0);
				val += val_to_sum;
				val_to_sum = (0 <= sample_rd_h - i && sample_rd_h - i < data_height && 
										     0 <= sample_lu_w && sample_lu_w < data_width) ?
						     data[(data_ch * data_height + sample_rd_h - i) * data_width + sample_lu_w] : static_cast<DType>(0);
				val += val_to_sum;
			}
			val = val / static_cast<DType>(4 * (sample_size - 1));  // average
			out[index] = val;
		}
	}
}


template<typename DType>
__global__ void RadSamBackwardKernel(
	int n, const DType* dOut,
	const int data_height, const int data_width, const int out_height, const int out_width,
	const int pad_h, const int pad_w, const int num_group, const int group_size,
	DType* dData) {
	// each kernel is for each element of data
	CUDA_KERNEL_LOOP(index, n) {
		int data_w = index % data_width;
		int data_h = index / data_width % data_height;
		int data_ch = index / data_width / data_height;

		if (data_ch >= num_group * group_size) {
			dData[index] = 0;
		} else {
			int out_w = data_w + pad_w - (num_group - 1);
			int out_h = data_h + pad_h - (num_group - 1);
			int out_ch = data_ch;

			int group_idx = out_ch / group_size;
			int sample_size = 2 * group_idx + 1;

			if (sample_size == 1) {
				dData[index] = (0 <= out_h && out_h < out_height && 0 <= out_w && out_w < out_width) ?
							   dOut[(out_ch * out_height + out_h) * out_width + out_w] : static_cast<DType>(0);
			} else {
				int outc_lu_w = out_w - group_idx;
				int outc_lu_h = out_h - group_idx;
				int outc_rd_w = out_w + group_idx;
				int outc_rd_h = out_h + group_idx;
				DType val = 0;

				for (int i = 0; i < sample_size - 1; ++i) {
					DType val_to_sum = (0 <= outc_lu_h && outc_lu_h < out_height && 
														   0 <= outc_lu_w + i && outc_lu_w + i < out_width) ?
									   dOut[(out_ch * out_height + outc_lu_h) * out_width + outc_lu_w + i] : static_cast<DType>(0);
					val += val_to_sum;
					val_to_sum = (0 <= outc_lu_h + i && outc_lu_h + i < out_height && 
												     0 <= outc_rd_w && outc_rd_w < out_width) ?
							     dOut[(out_ch * out_height + outc_lu_h + i) * out_width + outc_rd_w] : static_cast<DType>(0);
					val += val_to_sum;
					val_to_sum = (0 <= outc_rd_h && outc_rd_h < out_height && 
												     0 <= outc_rd_w - i && outc_rd_w - i < out_width) ?
							     dOut[(out_ch * out_height + outc_rd_h) * out_width + outc_rd_w - i] : static_cast<DType>(0);
					val += val_to_sum;
					val_to_sum = (0 <= outc_rd_h - i && outc_rd_h - i < out_height && 
												     0 <= outc_lu_w && outc_lu_w < out_width) ?
							     dOut[(out_ch * out_height + outc_rd_h - i) * out_width + outc_lu_w] : static_cast<DType>(0);
					val += val_to_sum;
				}
				val = val / static_cast<DType>(4 * (sample_size - 1));  // average
				dData[index] = val; 
			}
		}
	}
}

}  // namespace cuda

template<typename DType>
inline void RadiateSampleForward(
	const Tensor<gpu, 4, DType>& data, const Tensor<gpu, 4, DType>& out,
	int pad_h, int pad_w, int num_group) {
	// get the shape of data and out
	int num = data.size(0);
	int data_height = data.size(2);
	int data_width = data.size(3);
	
	int out_channels = out.size(1);
	int out_height = out.size(2);
	int out_width = out.size(3);

	int group_size = data.size(1) / num_group;

	// lauch cuda kernel
	int num_threads = out_channels * out_height * out_width;

	for (int n = 0; n < num; ++n) {
		Tensor<gpu, 3, DType> input_3d = data[n];
        Tensor<gpu, 3, DType> output_3d = out[n];

		cuda::RadSamForwardKernel<DType>
			<<<mxnet::op::mxnet_op::cuda_get_num_blocks(num_threads), cuda::kBaseThreadNum,
			   0, Stream<gpu>::GetStream(output_3d.stream_)>>>(
			num_threads, input_3d.dptr_, data_height, data_width, out_height, out_width,
			pad_h, pad_w, num_group, group_size, output_3d.dptr_);
		MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::RadSamForwardKernel);
	}
}


template<typename DType>
inline void RadiateSampleBackward(
	const Tensor<gpu, 4, DType>& dData, const Tensor<gpu, 4, DType>& dOut,
	int pad_h, int pad_w, int num_group) {
	//
	int num = dData.size(0);
	int data_channels = dData.size(1);
	int data_height = dData.size(2);
	int data_width = dData.size(3);
	
	// int out_channels = dOut.size(1);
	int out_height = dOut.size(2);
	int out_width = dOut.size(3);

	int group_size = data_channels / num_group;

	// lauch cuda kernel
	int num_threads = data_channels * data_height * data_width;

	for (int n = 0; n < num; ++n) {
		Tensor<gpu, 3, DType> in_grad_3d = dData[n];
		Tensor<gpu, 3, DType> out_grad_3d = dOut[n];

		cuda::RadSamBackwardKernel<DType>
			<<<mxnet::op::mxnet_op::cuda_get_num_blocks(num_threads), cuda::kBaseThreadNum,
			   0, Stream<gpu>::GetStream(in_grad_3d.stream_)>>>(
			num_threads, out_grad_3d.dptr_, data_height, data_width, out_height, out_width,
			pad_h, pad_w, num_group, group_size, in_grad_3d.dptr_);
		MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::RadSamBackwardKernel);
	}
}

}  // namespace mshadow


namespace mxnet {
namespace op {

NNVM_REGISTER_OP(RadiateSample)
.set_attr<FCompute>("FCompute<gpu>", RadiateSampleCompute<gpu>);

NNVM_REGISTER_OP(_backward_RadiateSample)
.set_attr<FCompute>("FCompute<gpu>", RadiateSampleGradCompute<gpu>);

}
}
