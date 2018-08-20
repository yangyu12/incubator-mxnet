#ifndef MXNET_OPERATOR_RADIATE_SAMPLE_INL_H_
#define MXNET_OPERATOR_RADIATE_SAMPLE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "../operator_common.h"
#include "./im2col.h"

namespace mxnet {
namespace op {
namespace radsam {
enum RadiateSampleOpInputs {kData};
enum RadiateSampleOpOutputs {kOut};
}  // namespace radsam


struct RadiateSampleParam : public dmlc::Parameter<RadiateSampleParam> {
    TShape pad;

    uint32_t num_group;

    DMLC_DECLARE_PARAMETER(RadiateSampleParam){
        DMLC_DECLARE_FIELD(pad).set_default(TShape())
        .describe("RadiateSample pad: (h, w). Defaults to (0, 0).");
        DMLC_DECLARE_FIELD(num_group).set_default(1)
        .describe("RadiateSample num_group: . Defaults to 1.");
    }

    bool operator==(const RadiateSampleParam& other) const {
        return this->pad == other.pad &&
               this->num_group == other.num_group;
    }
};


void RadiateSampleParamParser(nnvm::NodeAttrs* attrs);


template<typename xpu, typename DType>
class RadiateSampleOp {
  public:
    void Init(RadiateSampleParam p) {
        this->param_ = p;
    }

    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data) {
        using namespace mshadow;
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(out_data.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        Tensor<xpu, 4, DType> data = in_data[radsam::kData].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> out = out_data[radsam::kOut].get<xpu, 4, DType>(s);
        CHECK_EQ(data.CheckContiguous(), true);
        CHECK_EQ(out.CheckContiguous(), true);

        RadiateSampleForward(data, out, param_.pad[0], param_.pad[1], param_.num_group);
    }

    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad) {
        // TODO: complete this method
        using namespace mshadow;
        CHECK_EQ(out_grad.size(), 1);
        CHECK_EQ(in_data.size(), 1);
        CHECK_EQ(in_grad.size(), 1);
        Stream<xpu> *s = ctx.get_stream<xpu>();

        Tensor<xpu, 4, DType> dOut = out_grad[radsam::kOut].get<xpu, 4, DType>(s);
        Tensor<xpu, 4, DType> dData = in_grad[radsam::kData].get<xpu, 4, DType>(s);
        CHECK_EQ(dOut.CheckContiguous(), true);
        CHECK_EQ(dData.CheckContiguous(), true);

        if (kAddTo == req[radsam::kData] || kWriteTo == req[radsam::kData]) {
            if (kWriteTo == req[radsam::kData]) {
                dData = static_cast<DType>(0);
            }
            RadiateSampleBackward(dData, dOut, param_.pad[0], param_.pad[1], param_.num_group);
        }
    }

  // private:
    // OPTION: other aux methods.

  private:
    RadiateSampleParam param_;
    // OPTION: other aux variables
};  // class RadiateSampleOp


// This function will be registered as op forward function. 
template<typename xpu>
void RadiateSampleCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx, 
                          const std::vector<TBlob>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<TBlob>& outputs) {
    const RadiateSampleParam& param = nnvm::get<RadiateSampleParam>(attrs.parsed);
    MSHADOW_REAL_TYPE_SWITCH(inputs[radsam::kData].type_flag_, DType, {
        RadiateSampleOp<xpu, DType> op;
        op.Init(param);
        op.Forward(ctx, inputs, req, outputs);
    });
}


// This function will be registered as op backward function. 
template<typename xpu>
void RadiateSampleGradCompute(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx, 
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
    const RadiateSampleParam& param = nnvm::get<RadiateSampleParam>(attrs.parsed);
    std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
    const TBlob &out_grad = inputs[radsam::kOut];
    const std::vector<TBlob> &in_grad = outputs;

    MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
        RadiateSampleOp<xpu, DType> op;
        op.Init(param);
        op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_RADIATE_SAMPLE_INL_H_