#include "radiate_sample-inl.h" 

namespace mshadow {
template<typename DType>
inline void RadiateSampleForward(
    const Tensor<cpu, 4, DType>& data, const Tensor<cpu, 4, DType>& out,
    int pad_h, int pad_w, int num_group) {
    // NOT IMPLEMENT
    return;
}


template<typename DType>
inline void RadiateSampleBackward(
    const Tensor<cpu, 4, DType>& dData, const Tensor<cpu, 4, DType>& dOut,
    int pad_h, int pad_w, int num_group) {
    // NOT IMPLEMENT
    return;
}
}  // namespace mshadow

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(RadiateSampleParam);


// This function will be registered as shape inference method.
static bool RadiateSampleShape(const nnvm::NodeAttrs& attrs,
                               std::vector<TShape> *in_shape,
                               std::vector<TShape> *out_shape) {
    using namespace mshadow;
    const RadiateSampleParam& param_ = nnvm::get<RadiateSampleParam>(attrs.parsed);

    // check whether in_shape
    CHECK_EQ(in_shape->size(), 1U);
    CHECK_EQ(out_shape->size(), 1U);
    out_shape->resize(1, TShape());
    const TShape &dshp = (*in_shape)[radsam::kData];
    CHECK_EQ(dshp.ndim(), 4) \
        << "Input data should be 4D in batch-num_filter-y-x";

    // infer output shape
    Shape<4> oshape;
    oshape[0] = dshp[0];
    int num_dropch = dshp[1] % param_.num_group;
    if (num_dropch > 0) { 
        printf("The number of channels %d is not multiple of num_group %d\n", dshp[1], param_.num_group);  // TODO: replace it with mxnet logging system.
    }
    oshape[1] = dshp[1] - num_dropch;
    oshape[2] = dshp[2] ? dshp[2] + 2 * param_.pad[0] - 2 * (param_.num_group - 1) : 0;
    oshape[3] = dshp[3] ? dshp[3] + 2 * param_.pad[1] - 2 * (param_.num_group - 1) : 0;
    SHAPE_ASSIGN_CHECK(*out_shape, radsam::kOut, oshape);

    return true;
}


// This function will be registered as type inference method.
static bool RadiateSampleType(const nnvm::NodeAttrs& attrs,
                              std::vector<int> *in_type,
                              std::vector<int> *out_type) {
    CHECK_EQ(in_type->size(), 1U);
    int dtype = (*in_type)[radsam::kData];
    CHECK_NE(dtype, -1) << "First input must have specific type";

    out_type->clear();
    out_type->push_back(dtype);

    return true;
}


// This function will be registered as parameter parsing method.
void RadiateSampleParamParser(nnvm::NodeAttrs* attrs) {
    using namespace mshadow;
    RadiateSampleParam param_;
    try {
        param_.Init(attrs->dict);
    } catch (const dmlc::ParamError& e) {
        std::ostringstream os;
        os << e.what();
        os << ", in operator " << attrs->op->name << "("
           << "name=\"" << attrs->name << "\"";
        for (const auto& k : attrs->dict) {
            os << ", " << k.first << "=\"" << k.second << "\"";
        }
        os << ")";
        throw dmlc::ParamError(os.str());
    }

    // set the default value. e.g.
    if (param_.pad.ndim() == 0) {
        param_.pad = Shape2(0, 0);
    } else {
        CHECK_EQ(param_.pad.ndim(), 2) 
            << "Only support 2-D sampling, but get" 
            << param_.pad.ndim() << "dimensions";
    }

    attrs->parsed = std::move(param_);
}


// This struct will be registered as gradient node making function
struct RadiateSampleGrad {
    const char *op_name;
    std::vector<nnvm::NodeEntry> operator()(const nnvm::NodePtr& n,
                                            const std::vector<nnvm::NodeEntry>& ograds) const {
        std::vector<nnvm::NodeEntry> heads(ograds.begin(), ograds.end());
        heads.push_back(n->inputs[radsam::kData]);
        return MakeGradNode(op_name, n, heads, n->attrs.dict);
    }
};


// register forward op
NNVM_REGISTER_OP(RadiateSample)
.describe(R"code(
    op document filled in this space...
)code" ADD_FILELINE)
.set_num_inputs(1)  // TODO: set number of inputs. Or replace x with an anonymous function [](const NodeAttrs& attrs) {}
.set_num_outputs(1) 
.set_attr_parser(RadiateSampleParamParser)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"output"};
})
.set_attr<nnvm::FInferShape>("FInferShape", RadiateSampleShape)
.set_attr<nnvm::FInferType>("FInferType", RadiateSampleType)
.set_attr<FCompute>("FCompute<cpu>", RadiateSampleCompute<cpu>)
.set_attr<nnvm::FGradient>("FGradient", RadiateSampleGrad{"_backward_RadiateSample"})
.add_argument("data", "NDArray-or-Symbol", "description about data should be filled in.")  // TODO: add more arguments if needed
.add_arguments(RadiateSampleParam::__FIELDS__());


// regiser backward op
NNVM_REGISTER_OP(_backward_RadiateSample)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr_parser(RadiateSampleParamParser)
.set_attr<FCompute>("FCompute<cpu>", RadiateSampleGradCompute<cpu>);

}  // namespace mxnet
}  // namespace op