#include <mshadow/tensor.h>
#include "./newreorg-inl.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(NewReorg)
.set_attr<FCompute>("FCompute<gpu>", NewReorgCompute<gpu>);

NNVM_REGISTER_OP(_backward_NewReorg)
.set_attr<FCompute>("FCompute<gpu>", NewReorgGradCompute<gpu>);

}
}
