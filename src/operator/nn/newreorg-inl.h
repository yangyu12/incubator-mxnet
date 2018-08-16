#ifndef MXNET_OPERATOR_REORG_INL_H_
#define MXNET_OPERATOR_REORG_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "../operator_common.h"
#include "./im2col.h"

namespace mxnet {
namespace op {
// enumeration
namespace newreorg {
enum NewReorgOpInputs {kData};
enum NewReorgOpOutputs {kOut};
}  // namespace newreorg

struct NewReorgParam : public dmlc::Parameter<NewReorgParam> {
    TShape kernel;
    TShape stride;
    TShape dilate;
    TShape pad;

    DMLC_DECLARE_PARAMETER(NewReorgParam){
        DMLC_DECLARE_FIELD(kernel).describe("newreorg kernel size: (h, w)");
        DMLC_DECLARE_FIELD(stride).set_default(TShape())
        .describe("newreorg stride: (h, w). Defaults to 1 for each dimension.");
        DMLC_DECLARE_FIELD(dilate).set_default(TShape())
        .describe("newreorg dilate: (h, w). Defaults to 1 for each dimension.");
        DMLC_DECLARE_FIELD(pad).set_default(TShape())
        .describe("zero pad for newreorg: (h, w). Defaults to no padding.");
    }

    // Adjust kernel size for effects of dilation in the dimension 'dim'.

    bool operator==(const NewReorgParam& other) const {
        return this->kernel == other.kernel &&
               this->stride == other.stride &&
               this->dilate == other.dilate &&
               this->pad == other.pad;
    }
};

void NewReorgParamParser(nnvm::NodeAttrs* attrs);

template<typename xpu, typename DType>
class NewReorgOp {
  public:
    void Init(NewReorgParam p) {
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
        LayerSetUp(in_data[newreorg::kData].shape_, out_data[newreorg::kOut].shape_);

        TShape fake_shape(out_data[newreorg::kData].shape_.ndim() - 1);
        for (index_t i = 0; i < fake_shape.ndim(); ++i) {
            fake_shape[i] = out_data[newreorg::kData].shape_[i+1];
        }

        for (index_t n = 0; n < num_; ++n) {
            im2col(s, in_data[newreorg::kData].dptr<DType>()+n*input_dim_, in_data[newreorg::kData].shape_,
                   fake_shape, param_.kernel, param_.pad, param_.stride, param_.dilate,
                   out_data[newreorg::kOut].dptr<DType>()+n*output_dim_);
        }
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
        LayerSetUp(in_grad[newreorg::kData].shape_, out_grad[newreorg::kOut].shape_);

        TShape fake_shape(out_grad[newreorg::kData].shape_.ndim() - 1);
        for (index_t i = 0; i < fake_shape.ndim(); ++i) {
            fake_shape[i] = out_grad[newreorg::kData].shape_[i+1];
        }

        for (index_t n = 0; n < num_; ++n) {
            col2im(s, out_grad[newreorg::kOut].dptr<DType>()+n*output_dim_, in_grad[newreorg::kData].shape_,
                   fake_shape, param_.kernel, param_.pad, param_.stride, param_.dilate,
                   in_grad[newreorg::kData].dptr<DType>()+n*input_dim_, req[newreorg::kData]);
        }
    }

  private:
    void LayerSetUp(const TShape& ishape, const TShape& oshape) {
        num_ = ishape[0];
        input_dim_ = ishape.ProdShape(1, ishape.ndim());
        output_dim_ = oshape.ProdShape(1, oshape.ndim());
    }

  private:
    NewReorgParam param_;

    index_t num_;
    index_t input_dim_;
    index_t output_dim_;
};  // class NewReorgOp

template<typename xpu>
void NewReorgCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx, const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
    const NewReorgParam& param = nnvm::get<NewReorgParam>(attrs.parsed);
    MSHADOW_REAL_TYPE_SWITCH(inputs[newreorg::kData].type_flag_, DType, {
        NewReorgOp<xpu, DType> op;
        op.Init(param);
        op.Forward(ctx, inputs, req, outputs);
    });
}

template<typename xpu>
void NewReorgGradCompute(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx, const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
    const NewReorgParam& param = nnvm::get<NewReorgParam>(attrs.parsed);
    std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
    const TBlob &out_grad = inputs[newreorg::kOut];
    const std::vector<TBlob> &in_grad = outputs;

    MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
        NewReorgOp<xpu, DType> op;
        op.Init(param);
        op.Backward(ctx, std::vector<TBlob>{out_grad}, in_data, req, in_grad);
    });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_REORG_INL_H_