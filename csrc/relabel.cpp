#include <Python.h>
#include <torch/script.h>

#include "cpu/relabel_cpu.h"

#ifdef _WIN32
PyMODINIT_FUNC PyInit__relabel(void) { return NULL; }
#endif

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
relabel_one_hop(torch::Tensor rowptr, torch::Tensor col,
                torch::optional<torch::Tensor> optional_value,
                torch::Tensor idx, bool bipartite) {
  if (rowptr.device().is_cuda()) {
#ifdef WITH_CUDA
    AT_ERROR("No CUDA version supported");
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return relabel_one_hop_cpu(rowptr, col, optional_value, idx, bipartite);
  }
}

static auto registry = torch::RegisterOperators().op(
    "torch_geometric_autoscale::relabel_one_hop", &relabel_one_hop);
