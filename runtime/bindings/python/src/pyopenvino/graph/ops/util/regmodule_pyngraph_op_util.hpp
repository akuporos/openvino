// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include "arithmetic_reduction.hpp"
#include "binary_elementwise_arithmetic.hpp"
#include "binary_elementwise_comparison.hpp"
#include "binary_elementwise_logical.hpp"
#include "index_reduction.hpp"
#include "op_annotations.hpp"
#include "unary_elementwise_arithmetic.hpp"

namespace py = pybind11;

void regmodule_op_util(py::module m);
