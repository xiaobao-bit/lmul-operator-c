#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  
#include "operator.hpp"     
namespace py = pybind11;

PYBIND11_MODULE(lmul, m) {
    m.doc() = "L-Mul operator lib";

    m.def("fp32_ele_downcast", &operands::fp32_ele_downcast, "Downcast FP32 element to FP8 element");
    m.def("fp32_mat_downcast", &operands::fp32_mat_downcast, "Downcast FP32 matrix to FP8 matrix");
    m.def("lmul_matmul", &operands::lmul_matmul, "Perform LMUL-based matrix multiplication");
}