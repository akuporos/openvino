// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_Type(py::module m);
void regclass_Bool(py::module m);
void regclass_Float32(py::module m);
void regclass_Float64(py::module m);
void regclass_Int8(py::module m);
// void regclass_Int16(py::module m);
void regclass_Int32(py::module m);
void regclass_Int64(py::module m);
void regclass_UInt8(py::module m);
// void regclass_UInt16(py::module m);
void regclass_UInt32(py::module m);
void regclass_UInt64(py::module m);
void regclass_BFloat16(py::module m);
