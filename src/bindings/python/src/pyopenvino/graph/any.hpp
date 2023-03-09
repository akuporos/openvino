// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "pyopenvino/graph/any.hpp"

namespace py = pybind11;


// Proxy base class
class PyAnyBase{
};

template <typename T>
py::type get_class_pytype() {
    auto tmp = T();
    py::object obj = py::cast(tmp);
    return py::type::of(obj);
}

// Proxy class for actual template
class PyAny : public ov::Any, public PyAnyBase {
};

// B - base class, "parent"
// C - templated class
// T - args...
template <typename B, template <typename T> typename C, typename T>
py::class_<C<T>> wrap(py::module m, std::string type_name) {
    py::class_<C<T>, B> cls(m, type_name.c_str());

    cls.def(py::init<>());

    cls.def("show", [](C<T> &self, const T &arg)
            { self.show(arg); });

    return std::move(cls); // return allows us to update class later
}

void regclass_graph_Any(py::module m);
