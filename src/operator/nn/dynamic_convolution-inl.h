
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
 * \file convolution-inl.h
 * \brief
 * \ref: https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
 * \author Bing Xu, Jun Wu, Da Zheng
*/
#ifndef MXNET_OPERATOR_NN_DYNAMIC_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_DYNAMIC_CONVOLUTION_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../linalg.h"
// #include "./im2col.h"


namespace mxnet {
namespace op {

namespace dynconv {
enum DynamicConvolutionOpInputs {kData, kAWeight, kWWeight};  // no bias
enum DynamicConvolutionOpOutputs {kOut};
enum DynamicConvolutionOpResource {kTempSpace};
enum DynamicConvolutionOpCudnnTune {kOff, kLimited, kFastest};
}  // namespace dynconv

struct DynamicConvolutionParam : public dmlc::Parameter<DynamicConvolutionParam> {
  TShape kernel;
  TShape stride;  // NOT SUPPORT stride != 1
  TShape dilate;
  TShape pad;  // the padding mechanism is not completed
  TShape sample;
  TShape s_stride;  // sample stride
  uint32_t num_filter;
  uint32_t num_group;  // NOT SUPPORT num_group != 1
  uint64_t workspace;
  // bool no_bias;
  dmlc::optional<int> cudnn_tune;
  bool cudnn_off;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(DynamicConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Dynamic convolution kernel size: (w,), (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .describe("Dynamic convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
    .describe("Dynamic convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(sample).set_default(TShape())
    .describe("Dynamic convolution sample: (w,), (h, w) or (d, h, w). Defaults to x for each dimension");
    DMLC_DECLARE_FIELD(s_stride).set_default(TShape())
    .describe("Dynamic convolution sampling stride: (w,), (h, w) or (d, h, w). Defaults to x for each dimension");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
    .describe("Convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
    .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
    .describe("Maximum temporary workspace allowed (MB) in convolution."
              "This parameter has two usages. When CUDNN is not used, it determines the "
              "effective batch size of the convolution kernel. When CUDNN is used, it controls "
              "the maximum temporary storage used for tuning the best CUDNN kernel when "
              "`limited_workspace` strategy is used.");
    // DMLC_DECLARE_FIELD(no_bias).set_default(false)
    // .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(cudnn_tune)
    .add_enum("off", dynconv::kOff)
    .add_enum("limited_workspace", dynconv::kLimited)
    .add_enum("fastest", dynconv::kFastest)
    .set_default(dmlc::optional<int>())
        .describe("Whether to pick convolution algo by running performance test.");
    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn for this layer.");
    DMLC_DECLARE_FIELD(layout)
    .add_enum("NCW", mshadow::kNCW)
    .add_enum("NCHW", mshadow::kNCHW)
    .add_enum("NCDHW", mshadow::kNCDHW)
    .add_enum("NHWC", mshadow::kNHWC)
    .add_enum("NDHWC", mshadow::kNDHWC)
    .set_default(dmlc::optional<int>())
    .describe("Set layout for input, output and weight. Empty for\n    "
              "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
  }
  // Adjusts kernel size for effects of dilation in the dimension `dim`.
  index_t DilatedKernelSize(int dim) const {
    return 1 + (kernel[dim] - 1) * dilate[dim];
  }

  bool operator==(const DynamicConvolutionParam& other) const {
    return this->kernel == other.kernel &&
           this->stride == other.stride &&
           this->dilate == other.dilate &&
           this->pad == other.pad &&
           this->sample == other.sample &&
           this->s_stride == other.s_stride &&
           this->num_filter == other.num_filter &&
           this->num_group == other.num_group &&
           this->workspace == other.workspace &&
           this->cudnn_tune == other.cudnn_tune &&
           this->cudnn_off == other.cudnn_off &&
           this->layout == other.layout;
  }
};

void DynamicConvolutionParamParser(nnvm::NodeAttrs* attrs);

typedef ParamOpSign<DynamicConvolutionParam> DynConvSignature;

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::DynamicConvolutionParam> {
  size_t operator()(const mxnet::op::DynamicConvolutionParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.dilate);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.sample);
    ret = dmlc::HashCombine(ret, val.s_stride);
    ret = dmlc::HashCombine(ret, val.num_filter);
    ret = dmlc::HashCombine(ret, val.num_group);
    ret = dmlc::HashCombine(ret, val.workspace);
    ret = dmlc::HashCombine(ret, val.cudnn_tune);
    ret = dmlc::HashCombine(ret, val.cudnn_off);
    ret = dmlc::HashCombine(ret, val.layout);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {
// declare function ...
// template<typename xpu, typename DType>
// void dynconv_im2col(mshadow::Stream<xpu>* s, const DType* data_im, const TShape& im_shape, const TShape& col_shape, const TShape& kernel_shape,
//     const TShape& pad, const TShape& stride, const TShape& dilation, const TShape& sample, const TShape& s_stride, DType* data_col);

// template<typename xpu, typename DType>
// void dynconv_col2im( mshadow::Stream<xpu>* s, const DType* data_col, const TShape& im_shape, const TShape& kernel_shape, const TShape& dilation,
//     const TShape& sample_shape, const TShape& s_stride, DType* data_im);

// template<typename xpu, typename DType>
// void dynconv_inprod(mshadow::Stream<xpu>* s, const DType* col_buffer, const DType* across_weight, const DType* within_weight, const TShape& im_shape,
//     const TShape& col_shape, const TShape& out_shape, const TShape& kernel_shape, const TShape& sample_shape, DType* out_data);

// template<typename xpu, typename DType>
// void dynconv_dAWeight(mshadow::Stream<xpu>* s, const DType* col_buffer, const DType* out_grad, const TShape& im_shape, const TShape& out_shape,
//     const TShape& kernel_shape, const TShape& sample_shape, DType* aweight_grad);

// template<typename xpu, typename DType>
// void dynconv_dWWeight(mshadow::Stream<xpu>* s, const DType* col_buffer, const DType* out_grad, const TShape& im_shape, const TShape& out_shape,
//     const TShape& kernel_shape, const TShape& sample_shape, DType* wweight_grad);

// template<typename xpu, typename DType>
// void dynconv_dCol(mshadow::Stream<xpu>* s, const DType* across_weight, const DType* within_weight, const DType* out_grad,
//     const TShape& im_shape, const TShape& out_shape, const TShape& kernel_shape, const TShape& sample_shape, DType* col_buffer);

template<typename xpu, typename DType>
class DynamicConvolutionOp {
 public:
  void Init(DynamicConvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    CHECK(param_.layout.value() == mshadow::kNCW ||
          param_.layout.value() == mshadow::kNCHW ||
          param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCW, NCHW and NCDHW layout";
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[dynconv::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req[dynconv::kOut], kWriteTo);
    LayerSetUp(in_data[dynconv::kData].shape_, out_data[dynconv::kOut].shape_);
    Stream<xpu>* s = ctx.get_stream<xpu>();

    // initialize weight and col_buffer 3D tensors  TODO: modify this
    // index_t M = conv_out_channels_ / group_;
    // index_t N = conv_out_spatial_dim_;
    // index_t K = kernel_dim_;
    // Tensor<xpu, 4, DType> output_4d = out_data[dynconv::kOut].get_with_shape<xpu, 4, DType>(
    //   Shape4(num_, group_, M, N), s);

    // allocate workspace for col_buffer
    Tensor<xpu, 1, DType> workspace = ctx.requested[dynconv::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
      // calculate the shape of col_buffer: Cin*k*k, H', W', s, s
    TShape col_buffer_shape(num_spatial_axes_ + 3);
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    for (index_t i = 1; i < col_buffer_shape.ndim() - 2; ++i) {
      col_buffer_shape[i] = out_data[0].shape_[i+1];
    }
    col_buffer_shape[col_buffer_shape.ndim() - 2] = param_.sample[0];
    col_buffer_shape[col_buffer_shape.ndim() - 1] = param_.sample[1];
      // create a column buffer using workspace and col_buffer_shape
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    for (index_t n = 0; n < num_; ++n) {
      // transform image to col_buffer
      dynconv_im2col(s, in_data[dynconv::kData].dptr<DType>()+n*input_dim_, channels_, out_data[0].shape_[2], out_data[0].shape_[3],
                     param_.kernel[0], param_.kernel[1], param_.dilate[0], param_.dilate[1],
                     param_.sample[0], param_.sample[1], param_.s_stride[0], col_buffer.dptr<DType>());
      dynconv_inprod(s, col_buffer.dptr<DType>(), in_data[dynconv::kAWeight].dptr<DType>()+n*aweight_dim_,
                     in_data[dynconv::kWWeight].dptr<DType>()+n*wweight_dim_, conv_out_channels_, channels_, out_data[0].shape_[2],
                     out_data[0].shape_[3], param_.kernel[0], param_.kernel[1], param_.sample[0], param_.sample[1],
                     out_data[dynconv::kOut].dptr<DType>()+n*output_dim_);
    }
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob>& out_grad,
                const std::vector<TBlob>& in_data,
                const std::vector<OpReqType>& req,
                const std::vector<TBlob>& in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 3U);
    CHECK_EQ(in_grad.size(), 3U);
    CHECK_EQ(req.size(), 3U);
    CHECK_EQ(in_data[dynconv::kAWeight].CheckContiguous(), true);
    CHECK_EQ(in_data[dynconv::kWWeight].CheckContiguous(), true);
    LayerSetUp(in_grad[dynconv::kData].shape_, out_grad[dynconv::kOut].shape_);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    // allocate workspace for col_buffer
    Tensor<xpu, 1, DType> workspace = ctx.requested[dynconv::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(col_buffer_size_), s);
      // calculate the shape of col_buffer: Cin*k*k, H', W', s, s
    TShape col_buffer_shape(num_spatial_axes_ + 3);
    col_buffer_shape[0] = conv_in_channels_ * param_.kernel.Size();
    for (index_t i = 1; i < col_buffer_shape.ndim() - 2; ++i) {
      col_buffer_shape[i] = out_grad[0].shape_[i+1];
    }
    col_buffer_shape[col_buffer_shape.ndim() - 2] = param_.sample[0];
    col_buffer_shape[col_buffer_shape.ndim() - 1] = param_.sample[1];
      // create a column buffer using workspace and col_buffer_shape
    TBlob col_buffer(workspace.dptr_, col_buffer_shape, xpu::kDevMask, DataType<DType>::kFlag);

    for (index_t n = 0; n < num_; ++n) {
      // gradient w.r.t. dynamic weight
      dynconv_im2col(s, in_data[dynconv::kData].dptr<DType>()+n*input_dim_, channels_, out_grad[0].shape_[2], out_grad[0].shape_[3],
                     param_.kernel[0], param_.kernel[1], param_.dilate[0], param_.dilate[1],
                     param_.sample[0], param_.sample[1], param_.s_stride[0], col_buffer.dptr<DType>());
      dynconv_dAWeight(s, col_buffer.dptr<DType>(), out_grad[dynconv::kOut].dptr<DType>()+n*output_dim_,
                       conv_out_channels_, channels_, out_grad[0].shape_[2], out_grad[0].shape_[3],
                       param_.kernel[0], param_.kernel[1], param_.sample[0], param_.sample[1],
                       in_grad[dynconv::kAWeight].dptr<DType>()+n*aweight_dim_);
      dynconv_dWWeight(s, col_buffer.dptr<DType>(), out_grad[dynconv::kOut].dptr<DType>()+n*output_dim_,
                       conv_out_channels_, channels_, out_grad[0].shape_[2], out_grad[0].shape_[3],
                       param_.kernel[0], param_.kernel[1], param_.sample[0], param_.sample[1],
                       in_grad[dynconv::kWWeight].dptr<DType>()+n*wweight_dim_);

      // gradient w.r.t. input data
      dynconv_dCol(s, in_data[dynconv::kAWeight].dptr<DType>()+n*aweight_dim_, in_data[dynconv::kWWeight].dptr<DType>()+n*wweight_dim_,
                   out_grad[dynconv::kOut].dptr<DType>()+n*output_dim_, conv_out_channels_, channels_, out_grad[0].shape_[2], out_grad[0].shape_[3],
                   param_.kernel[0], param_.kernel[1], param_.sample[0], param_.sample[1], col_buffer.dptr<DType>());
      dynconv_col2im(s, col_buffer.dptr<DType>(), channels_, out_grad[0].shape_[2], out_grad[0].shape_[3],
                     param_.kernel[0], param_.kernel[1], param_.dilate[0], param_.dilate[1], param_.sample[0], param_.sample[1],
                     param_.s_stride[0], in_grad[dynconv::kData].dptr<DType>()+n*input_dim_);
    }
  }

 private:
  void LayerSetUp(const TShape& ishape, const TShape& oshape) {
    channel_axis_ = 1;  // hard code channel axis
    const index_t first_spatial_axis = channel_axis_ + 1;
    const index_t num_axes = param_.kernel.ndim() + 2;
    num_spatial_axes_ = num_axes - first_spatial_axis;

    // TODO: considering to deprecate is_1x1_ flag
    is_1x1_ = true;
    for (index_t i = 0; i < param_.kernel.ndim(); ++i) {
      is_1x1_ &= param_.kernel[i] == 1 && param_.stride[i] == 1 && param_.pad[i] == 0;
      if (!is_1x1_) break;
    }

    // batch size
    num_ = ishape[0];
    // number of input channels
    channels_ = ishape[1];
    group_ = param_.num_group;
    conv_out_channels_ = param_.num_filter;
    conv_in_channels_ = channels_;
    kernel_dim_ = conv_in_channels_ / group_ * param_.kernel.Size();
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    conv_out_spatial_dim_ = oshape.ProdShape(2, oshape.ndim());
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_ * param_.sample.Size();  // Cin*k*k * H' * W' * Sh * Sw
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;  // Cout * H' * W'
    // size of the column buffer used for storing im2col-ed pixels
    col_buffer_size_ = kernel_dim_ * group_ * conv_out_spatial_dim_ * param_.sample.Size();  // Cin*k*k * H' * W' * Sh * Sw
    // input/output image size (#channels * height * width)
    input_dim_ = ishape.ProdShape(1, ishape.ndim());
    output_dim_ = oshape.ProdShape(1, oshape.ndim());
    aweight_dim_ = conv_in_channels_ * oshape.ProdShape(2, oshape.ndim());
    wweight_dim_ = param_.kernel.Size() * oshape.ProdShape(2, oshape.ndim());
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;  // TODO: revisit
    num_kernels_col2im_ = input_dim_;  // TODO: revisit
  }

 private:
  DynamicConvolutionParam param_;
  index_t channel_axis_;  // channel axis of the input (# 1)
  index_t channels_;  // number of channels of input image (# Cin)
  index_t num_spatial_axes_;  // number of spatial axes (# 1, 2 or 3)
  index_t num_;  // batch size (# B)
  index_t group_;  // number of groups (# G)
  index_t conv_out_channels_;  // number of output channels (num_filter) (# Cout)
  index_t conv_out_spatial_dim_;  // number of pixels of output images per channel (# H' * W')
  index_t conv_in_channels_;  // number of input channels (# Cin)
  index_t kernel_dim_;  // number of input channels per group * kernel size (# Cin * k * k)
  index_t weight_offset_;  // number of output channels per group * kernel_dim_ (# Cout * Cin * k * k)
  index_t col_offset_;  // (# Cin*k*k * H' * W' * s * s)
  index_t output_offset_;  // (# Cout * H' * W')
  index_t col_buffer_size_;  // (# Cin*k*k * H' * W' * Sh * Sw)
  index_t input_dim_;  // (# Cin * H * W)
  index_t aweight_dim_;  // (# Cin * H' * W')
  index_t wweight_dim_;  // (# k*k * H' * W')
  index_t output_dim_;  // (# Cout * H' * W')
  index_t num_kernels_im2col_;
  index_t num_kernels_col2im_;
  bool is_1x1_;
};  // class DynamicConvolutionOp

template<typename xpu>
void DynamicConvolutionCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  const DynamicConvolutionParam& param = nnvm::get<DynamicConvolutionParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[dynconv::kData].type_flag_, DType, {
    DynamicConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void DynamicConvolutionGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  const DynamicConvolutionParam& param = nnvm::get<DynamicConvolutionParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  const TBlob &out_grad = inputs[0];
  const std::vector<TBlob> &in_grad = outputs;

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    DynamicConvolutionOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
  });
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_NN_DYNAMIC_CONVOLUTION_INL_H_
