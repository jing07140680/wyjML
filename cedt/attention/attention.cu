#include <torch/extension.h>
#include <vector>

template <typename T>
std::vector<torch::Tensor> attention_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
    CHECK_INPUT(query);
    CHECK_INPUT(key);
    CHECK_INPUT(value);
    if (query.scalar_type() == torch::kFloat) {
        return attention_forward_cuda<float>(query, key, value);
    } else if (query.scalar_type() == torch::kDouble) {
        return attention_forward_cuda<double>(query, key, value);
    } else {
        AT_ERROR("Unsupported tensor type");
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward, "Attention forward (CUDA)");
}
