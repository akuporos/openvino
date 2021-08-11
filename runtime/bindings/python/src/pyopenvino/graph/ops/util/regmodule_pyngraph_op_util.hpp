// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include "graph/ops/util/arithmetic_reduction.hpp"
#include "graph/ops/util/binary_elementwise_arithmetic.hpp"
#include "graph/ops/util/binary_elementwise_comparison.hpp"
#include "graph/ops/util/binary_elementwise_logical.hpp"
#include "graph/ops/util/index_reduction.hpp"
#include "graph/ops/util/op_annotations.hpp"
#include "graph/ops/util/unary_elementwise_arithmetic.hpp"

namespace py = pybind11;

void regmodule_op_util(py::module m);
