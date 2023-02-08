
// #include <torch/script.h>
// #include <torch/custom_class.h>

// #include <string>
// #include <vector>

#include "cpu/relabel2hop_cpu.cpp"


TORCH_LIBRARY(torch_geometric_autoscale, m) {
  m.class_<Relabel2hop<std::string>>("Relabel2hop")
    .def(torch::init<torch::Tensor,torch::Tensor,torch::optional<torch::Tensor>>())
    .def("relabel_one_hop", &Relabel2hop<std::string>::relabel_one_hop)
    .def("relabel_one_hop_compensation", &Relabel2hop<std::string>::relabel_one_hop_compensation)
    .def("relabel_one_hop_compensation_rm2hop", &Relabel2hop<std::string>::relabel_one_hop_compensation_rm2hop);
}

// TORCH_LIBRARY(torch_geometric_autoscale, m) {
//   m.class_<Relabel2hop<std::string>>("Relabel2hop")
//     .def(torch::init<>())
//     .def("relabel_one_hop_compensation", &Relabel2hop<std::string>::relabel_one_hop_compensation)
//     .def("relabel_one_hop_compensation_rm2hop", &Relabel2hop<std::string>::relabel_one_hop_compensation_rm2hop);
// }