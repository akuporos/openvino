// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pyopenvino/core/ie_infer_request.hpp"

#include <ie_common.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>

#include <string>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/ie_executable_network.hpp"
#include "pyopenvino/core/ie_preprocess_info.hpp"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<ov::runtime::ProfilingInfo>);

void regclass_InferRequest(py::module m) {
    py::class_<InferRequestWrapper, std::shared_ptr<InferRequestWrapper>> cls(m, "InferRequest");
    cls.def(
        "set_input",
        [](InferRequestWrapper& self, const py::dict& inputs) {
            Common::set_request_blobs(self._request, inputs);
        },
        py::arg("inputs"));

    cls.def(
        "set_output",
        [](InferRequestWrapper& self, const py::dict& results) {
            Common::set_request_blobs(self._request, results);
        },
        py::arg("results"));

    cls.def(
        "_infer",
        [](InferRequestWrapper& self, const py::dict& inputs) {
            // Update inputs if there are any
            if (!inputs.empty()) {
                Common::set_request_blobs(self._request, inputs);
            }
            // Call Infer function
            self._startTime = Time::now();
            self._request.infer();
            self._endTime = Time::now();
            // Get output Blobs and return
            Containers::PyResults results;
            for (auto& out : self._outputsInfo) {
                results[out.first] = self._request.get_tensor(out.first);
            }
            return results;
        },
        py::arg("inputs"));

    cls.def(
        "start_async",
        [](InferRequestWrapper& self, const py::dict inputs, py::object userdata) {
            py::gil_scoped_release release;
            if (!inputs.empty()) {
                Common::set_request_blobs(self._request, inputs);
            }
            TODO: check for None so next async infer userdata can be updated
            if (!userdata.empty())
            {
                if (user_callback_defined)
                {
                    self._request.SetCompletionCallback([self, userdata]() {
                        // py::gil_scoped_acquire acquire;
                        auto statusCode = const_cast<InferRequestWrapper&>(self).Wait(
                            InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
                        self._request.user_callback(self, statusCode, userdata);
                        // py::gil_scoped_release release;
                    });
                }
                else
                {
                    py::print("There is no callback function!");
                }
            }
            self._startTime = Time::now();
            self._request.start_async();
        },
        py::arg("inputs"),
        py::arg("userdata"));

    cls.def("cancel", [](InferRequestWrapper& self) {
        self._request.cancel();
    });

    cls.def(
        "wait",
        [](InferRequestWrapper& self) {
            py::gil_scoped_release release;
            return self._request.wait();
        });

    cls.def(
        "set_callback",
        [](InferRequestWrapper& self, py::function f_callback) {
            self._request.set_callback([&self, f_callback]() {
                self._endTime = Time::now();
                self._request.wait();

                // Acquire GIL, execute Python function
                py::gil_scoped_acquire acquire;
                f_callback(self);
            });
        },
        py::arg("f_callback"));


        cls.def(
        "set_completion_callback",
        [](InferRequestWrapper& self, py::function f_callback, py::object userdata) {
            self._request.SetCompletionCallback([&self, f_callback, userdata]() {
                self._endTime = Time::now();
                InferenceEngine::StatusCode statusCode =
                    self._request.Wait(InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
                if (statusCode == InferenceEngine::StatusCode::RESULT_NOT_READY) {
                    statusCode = InferenceEngine::StatusCode::OK;
                }
                // Acquire GIL, execute Python function
                py::gil_scoped_acquire acquire;
                f_callback(self, statusCode, userdata);
            });
        },
        py::arg("f_callback"),
        py::arg("userdata"));

    cls.def("get_profiling_info", [](InferRequestWrapper& self) {
        const std::vector<ov::runtime::ProfilingInfo> prof_vec = self._request.get_profiling_info();
        return prof_vec;
    });

    cls.def(
        "preprocess_info",
        [](InferRequestWrapper& self, const std::string& name) {
            return self._request.GetPreProcess(name);
        },
        py::arg("name"));

    cls.def("get_tensor", [](InferRequestWrapper& self, const std::string& name) {
        self._request.get_tensor(name);
        },
        py::arg("name"));

    cls.def("set_tensor", [](InferRequestWrapper& self, const std::string& name, const ov::runtime::Tensor& tensor) {
        self._request.set_tensor(name, tensor);
    },
    py::arg("name"),
    py::arg("tensor"));

    //    cls.def_property_readonly("preprocess_info", [](InferRequestWrapper& self) {
    //
    //    });

    cls.def_property_readonly("input_blobs", [](InferRequestWrapper& self) {
        Containers::PyResults input_blobs;
        for (auto& in : self._inputsInfo) {
            input_blobs[in.first] = self._request.get_tensor(in.first);
        }
        return input_blobs;
    });

    cls.def_property_readonly("output_blobs", [](InferRequestWrapper& self) {
        Containers::PyResults output_blobs;
        for (auto& out : self._outputsInfo) {
            output_blobs[out.first] = self._request.get_tensor(out.first);
        }
        return output_blobs;
    });

    cls.def_property_readonly("latency", [](InferRequestWrapper& self) {
        return self.getLatency();
    });
}
