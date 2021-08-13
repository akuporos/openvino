// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include "regmodule_pyngraph_types.hpp"

namespace py = pybind11;

void regmodule_types(py::module m)
{
    regclass_Type(m);
}
