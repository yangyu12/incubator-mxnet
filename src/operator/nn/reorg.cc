#include "reorg-inl.h"

namespace mshadow {
template<typename DType>
inline void ReorgForward(
    const Tensor<cpu, 4, DType> &data, const Tensor<cpu, 4, DType> &out,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int dilate_h, int dilate_w, int pad_h, int pad_w) {
    // NOT IMPLEMENTED
    return;
}

template<typename DType>
inline void ReorgBackward(
    const Tensor<cpu, 4, DType> dData, const Tensor<cpu, 4, DType> dOut,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int dilate_h, int dilate_w, int pad_h, int pad_w) {
    // NOT IMPLEMENTED
    return;
}

}  // namespace mshadow

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ReorgParam);

static bool ReorgShape(const nnvm::NodeAttrs& attrs,
                       std::vector<TShape> *in_shape,
                       std::vector<TShape> *out_shape) {
    using namespace mshadow;
    const ReorgParam& param_ = nnvm::get<ReorgParam>(attrs.parsed);
    out_shape->resize(1, TShape());
    const TShape &dshp = (*in_shape)[reorg::kData];
    if (dshp.ndim() ==  0) return false;

    CHECK_EQ(dshp.ndim(), 4U) \
        << "Input data should be 4D in batch-num_filter-y-x";
    Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);

    const index_t dilated_ksize_y = (param_.kernel[0] - 1) * param_.dilate[0] + 1;
    const index_t dilated_ksize_x = (param_.kernel[1] - 1) * param_.dilate[1] + 1;
    CHECK_GT(param_.kernel.Size(), 0U) \
        << "incorrect kernel size: " << param_.kernel;
    CHECK_GT(param_.stride.Size(), 0U) \
        << "incorrect stride size: " << param_.stride;
    CHECK_GT(param_.dilate.Size(), 0U) \
        << "incorrect dilate size: " << param_.dilate;
    CHECK_GT(param_.pad.Size(), 0U) \
        << "incorrect pad size: " << param_.pad;

    // output shape inference
    Shape<4> oshape;
    oshape[0] = dshape[0];
    oshape[1] = dshape[1] * param_.kernel[0] * param_.kernel[1];
    oshape[2] = dshape[2] ?
        (dshape[2] + 2 * param_.pad[0] - dilated_ksize_y) / param_.stride[0] + 1 : 0;
    oshape[3] = dshape[3] ?
        (dshape[3] + 2 * param_.pad[1] - dilated_ksize_x) / param_.stride[1] + 1 : 0;
    SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));

    // Perform incomplete shape inference. Fill in the missing values in data shape.
    // 1) We can always fill in the batch_size.
    // 2) We can back-calculate the input height/width if the corresponding stride is 1.
    oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
    dshape[0] = oshape[0];
    if (oshape[2] && param_.stride[0] == 1) {
        dshape[2] = oshape[2] + dilated_ksize_y - 1 - 2 * param_.pad[0];
    }
    if (oshape[3] && param_.stride[1] == 1) {
        dshape[3] = oshape[3] + dilated_ksize_x - 1 - 2 * param_.pad[1];
    }
    SHAPE_ASSIGN_CHECK(*in_shape, reorg::kData,
        ConvertLayout(dshape, kNCHW, param_.layout.value()));
    // Check whether the kernel sizes are valid
    if (dshape[2] != 0) {
        CHECK_LE(dilated_ksize_y, dshape[2] + 2 * param_.pad[0]) << "kernel size exceed input";
    }
    if (dshape[3] != 0) {
        CHECK_LE(dilated_ksize_x, dshape[3] + 2 * param_.pad[1]) << "kernel size exceed input";
    }
    return true;
}

static bool ReorgType(const nnvm::NodeAttrs& attrs,
                      std::vector<int> *in_type,
                      std::vector<int> *out_type) {
    const ReorgParam& param_ = nnvm::get<ReorgParam>(attrs.parsed);
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
        if ((*in_type)[i] == -1) {
            (*in_type)[i] = dtype;
        } else {
            UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments(param_)[i]);
        }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
}

struct ReorgGrad {
    const char *op_name;
    std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                            const std::vector<nnvm::NodeEntry>& ograds) const {
    const ReorgParam& param = nnvm::get<ReorgParam>(n->attrs.parsed);
    std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
    heads.push_back(n->inputs[reorg::kData]);
    return MakeGradNode(op_name, n, heads, n->attrs.dict);
    }
};


NNVM_REGISTER_OP(Reorg)
.describe(R"code(
    balabala

)code" ADD_FILELINE)
.set_num_inputs([](const NodeAttrs& attrs) {
  const ReorgParam& params = nnvm::get<ReorgParam>(attrs.parsed);
  return 1;
})
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
    const ReorgParam& params = nnvm::get<ReorgParam>(attrs.parsed);
    return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<nnvm::FInferShape>("FInferShape", ReorgShape)
.set_attr<nnvm::FInferType>("FInferType", ReorgType)
.set_attr<FCompute>("FCompute<cpu>", ReorgCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ReorgGrad{"_backward_Reorg"})
.add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionOp.")
.add_arguments(ReorgParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_Reorg)
.set_num_outputs([](const NodeAttrs& attrs) {
  const ReorgParam& params = nnvm::get<ReorgParam>(attrs.parsed);
  return 1;
})
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FCompute>("FCompute<cpu>", ReorgGradCompute<cpu>);

}  // namespace mxnet
}  // namespace op