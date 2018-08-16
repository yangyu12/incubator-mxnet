#ifndef MXNET_OPERATOR_REORG_INL_H_
#define MXNET_OPERATOR_REORG_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "../operator_common.h"

namespace mxnet {
namespace op {
// enumeration
namespace reorg {
enum ReorgOpInputs {kData};
enum ReorgOpOutputs {kOut};
}  // namespace reorg

struct ReorgParam : public dmlc::Parameter<ReorgParam> {
    TShape kernel;
    TShape stride;
    TShape dilate;
    TShape pad;

    DMLC_DECLARE_PARAMETER(ReorgParam){
        DMLC_DECLARE_FIELD(kernel).describe("Reorg kernel size: (h, w)");
        DMLC_DECLARE_FIELD(stride).set_default(TShape())
        .describe("Reorg stride: (h, w). Defaults to 1 for each dimension.");
        DMLC_DECLARE_FIELD(dilate).set_default(TShape())
        .describe("Reorg dilate: (h, w). Defaults to 1 for each dimension.");
        DMLC_DECLARE_FIELD(pad).set_default(TShape())
        .describe("zero pad for Reorg: (h, w). Defaults to no padding.");
    }

    // Adjust kernel size for effects of dilation in the dimension 'dim'.

    bool operator==(const ReorgParam& other) const {
        return this->kernel == other.kernel &&
               this->stride == other.stride &&
               this->dilate == other.dilate &&
               this->pad == other.pad;
    }
};

void ReorgParamParser(nnvm::NodeAttrs* attrs);

template<typename xpu, typename DType>
class ReorgOp {
  public:
    void Init(ReorgParam p) {
        this->param_ = p;
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data) {
        using namespace mshadow;
        // check a lot of info
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        Tensor<xpu, 4, DType> data = in_data[reorg::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> out = out_data[reorg::kOut].get<xpu, 4, DType>(s);
        CHECK_EQ(data.CheckContiguous(), true);
        CHECK_EQ(out.CheckContiguous(), true);

        ReorgForward(data, out, param_.kernel[0], param_.kernel[1], param_.stride[0], param_.stride[1],
                     param_.dilate[0], param_.dilate[1], param_.pad[0], param_.pad[1]);
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad) {
        using namespace mshadow;
        // check a lot of info
        CHECK_EQ(out_grad.size(), 1);
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(in_grad.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        Tensor<xpu, 4, DType> dData = in_grad[reorg::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> dOut = out_grad[reorg::kOut].get<xpu, 4, DType>(s);
        CHECK_EQ(dData.CheckContiguous(), true);
        CHECK_EQ(dOut.CheckContiguous(), true);

        // TODO: figure out the principle of req
        if (req[reorg::kData] == kWriteTo || req[reorg::kData] == kAddTo) {
            if (req[reorg::kData] == kWriteTo) {
                dData = static_cast<DType>(0);
            }
            ReorgBackward(dData, dOut, param_.kernel[0], param_.kernel[1], param_.stride[0], param_.stride[1],
                          param_.dilate[0], param_.dilate[1], param_.pad[0], param_.pad[1]);
        }
    }

  private:
    ReorgParam param_;
};  // class ReorgOp

template<typename xpu>
void ReorgCompute(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx, const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
    const ReorgParam& param = nnvm::get<ReorgParam>(attrs.parsed);
    MSHADOW_REAL_TYPE_SWITCH(inputs[reorg::kData].type_flag_, DType, {
        ReorgOp<xpu, DType> op;
        op.Init(param);
        op.Forward(ctx, inputs, req, outputs);
    });
}

template<typename xpu>
void ReorgGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx, const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
    const ReorgParam& param = nnvm::get<ReorgParam>(attrs.parsed);
    std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
    const TBlob &out_grad = inputs[reorg::kOut];
    const std::vector<TBlob> &in_grad = outputs;

    MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
        ReorgOp<xpu, DType> op;
        op.Init(param);
        op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_REORG_INL_H_