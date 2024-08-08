#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> attention_forward_cuda(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  return attention_forward_cuda(query, key, value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &attention_forward, "Attention forward (CUDA)");
}
