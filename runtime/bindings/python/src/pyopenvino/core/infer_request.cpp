// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pyopenvino/core/infer_request.hpp"

#include <ie_common.h>
#include <pybind11/functional.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include <string>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/core/containers.hpp"
#include "pyopenvino/core/executable_network.hpp"
#include "pyopenvino/core/ie_preprocess_info.hpp"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<ov::runtime::ProfilingInfo>);
PYBIND11_MAKE_OPAQUE(std::vector<InferenceEngine::VariableState>);

void regclass_InferRequest(py::module m) {
    py::class_<InferRequestWrapper, std::shared_ptr<InferRequestWrapper>> cls(m, "InferRequest");
    cls.def(
        "set_tensors",
        [](InferRequestWrapper& self, const Containers::TensorNameMap& inputs) {
            for (auto&& pair : inputs) {
                self._request.set_tensor(pair.first, pair.second);
            }
        },
        py::arg("inputs"));

    cls.def(
        "set_output_tensors",
        [](InferRequestWrapper& self, const Containers::TensorIndexMap& outputs) {
            for (auto&& pair : outputs) {
                self._request.set_output_tensor(pair.first, pair.second);
            }
        },
        py::arg("outputs"));

    cls.def(
        "set_input_tensors",
        [](InferRequestWrapper& self, const Containers::TensorIndexMap& inputs) {
            for (auto&& pair : inputs) {
                self._request.set_input_tensor(pair.first, pair.second);
            }
        },
        py::arg("inputs"));

    cls.def(
        "_infer",
        [](InferRequestWrapper& self, const Containers::TensorIndexMap& inputs) {
            // Update inputs if there are any
            if (!inputs.empty()) {
                for (auto&& pair : inputs) {
                    self._request.set_input_tensor(pair.first, pair.second);
                }
            }
            // Call Infer function
            self._startTime = Time::now();
            self._request.infer();
            self._endTime = Time::now();
            Containers::InferResults results;
            for (auto& out : self._outputs) {
                results.push_back(self._request.get_tensor(out));
            }
            return results;
        },
        py::arg("inputs"));

    cls.def(
        "_infer",
        [](InferRequestWrapper& self, const Containers::TensorNameMap& inputs) {
            // Update inputs if there are any
            if (!inputs.empty()) {
                for (auto&& pair : inputs) {
                    self._request.set_tensor(pair.first, pair.second);
                }
            }
            // Call Infer function
            self._startTime = Time::now();
            self._request.infer();
            self._endTime = Time::now();
            Containers::TensorNameMap results;
            for (auto& out : self._outputs) {
                results[out.get_any_name()] = self._request.get_tensor(out);
            }
            return results;
        },
        py::arg("inputs"));

    cls.def(
        "_start_async",
        [](InferRequestWrapper& self, const Containers::TensorIndexMap& inputs) {
            py::gil_scoped_release release;
            if (!inputs.empty()) {
               for (auto&& pair : inputs) {
                    self._request.set_input_tensor(pair.first, pair.second);
                }
            }
            // TODO: check for None so next async infer userdata can be updated
            // if (!userdata.empty())
            // {
            //     if (user_callback_defined)
            //     {
            //         self._request.SetCompletionCallback([self, userdata]() {
            //             // py::gil_scoped_acquire acquire;
            //             auto statusCode = const_cast<InferRequestWrapper&>(self).Wait(
            //                 InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
            //             self._request.user_callback(self, statusCode, userdata);
            //             // py::gil_scoped_release release;
            //         });
            //     }
            //     else
            //     {
            //         py::print("There is no callback function!");
            //     }
            // }
            self._startTime = Time::now();
            self._request.start_async();
        },
        py::arg("inputs"));

    cls.def(
        "_start_async",
        [](InferRequestWrapper& self, const Containers::TensorNameMap& inputs) {
            py::gil_scoped_release release;
            if (!inputs.empty()) {
               for (auto&& pair : inputs) {
                    self._request.set_tensor(pair.first, pair.second);
                }
            }
            // TODO: check for None so next async infer userdata can be updated
            // if (!userdata.empty())
            // {
            //     if (user_callback_defined)
            //     {
            //         self._request.SetCompletionCallback([self, userdata]() {
            //             // py::gil_scoped_acquire acquire;
            //             auto statusCode = const_cast<InferRequestWrapper&>(self).Wait(
            //                 InferenceEngine::IInferRequest::WaitMode::STATUS_ONLY);
            //             self._request.user_callback(self, statusCode, userdata);
            //             // py::gil_scoped_release release;
            //         });
            //     }
            //     else
            //     {
            //         py::print("There is no callback function!");
            //     }
            // }
            self._startTime = Time::now();
            self._request.start_async();
        },
        py::arg("inputs"));

    cls.def("cancel", [](InferRequestWrapper& self) {
        self._request.cancel();
    });

    cls.def("wait", [](InferRequestWrapper& self) {
        py::gil_scoped_release release;
        self._request.wait();
    });

    cls.def(
        "wait_for",
        [](InferRequestWrapper& self, const std::chrono::milliseconds timeout) {
            py::gil_scoped_release release;
            return self._request.wait_for(timeout);
        },
        py::arg("timeout"));

    cls.def(
        "set_callback",
        [](InferRequestWrapper& self, py::function f_callback) {
            self._request.set_callback([&self, f_callback](std::exception_ptr exceptionPtr) {
                self._endTime = Time::now();
                try {
                    if (exceptionPtr) {
                        std::rethrow_exception(exceptionPtr);
                    }
                } catch (const std::exception& e) {
                    IE_THROW() << "Caught exception: " << e.what();
                }
                // Acquire GIL, execute Python function
                py::gil_scoped_acquire acquire;
                f_callback(exceptionPtr);
            });
        },
        py::arg("f_callback"));

    cls.def("get_profiling_info", [](InferRequestWrapper& self) {
        const std::vector<ov::runtime::ProfilingInfo> prof_vec = self._request.get_profiling_info();
        return prof_vec;
    });

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const std::string& name) {
            return self._request.get_tensor(name);
        },
        py::arg("name"));

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const ov::Output<const ov::Node>& port) {
            return self._request.get_tensor(port);
        },
        py::arg("port"));

    cls.def(
        "get_tensor",
        [](InferRequestWrapper& self, const ov::Output<ov::Node>& port) {
            return self._request.get_tensor(port);
        },
        py::arg("port"));

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const std::string& name, const ov::runtime::Tensor& tensor) {
            self._request.set_tensor(name, tensor);
        },
        py::arg("name"),
        py::arg("tensor"));

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const ov::Output<const ov::Node>& port, const ov::runtime::Tensor& tensor) {
            self._request.set_tensor(port, tensor);
        },
        py::arg("port"),
        py::arg("tensor"));

    cls.def(
        "set_tensor",
        [](InferRequestWrapper& self, const ov::Output<ov::Node>& port, const ov::runtime::Tensor& tensor) {
            self._request.set_tensor(port, tensor);
        },
        py::arg("port"),
        py::arg("tensor"));

    cls.def(
        "set_input_tensor",
        [](InferRequestWrapper& self, size_t idx, const ov::runtime::Tensor& tensor) {
            self._request.set_input_tensor(idx, tensor);
        },
        py::arg("idx"),
        py::arg("tensor"));

    cls.def(
        "set_input_tensor",
        [](InferRequestWrapper& self, const ov::runtime::Tensor& tensor) {
            self._request.set_input_tensor(tensor);
        },
        py::arg("tensor"));

    cls.def(
        "set_output_tensor",
        [](InferRequestWrapper& self, size_t idx, const ov::runtime::Tensor& tensor) {
            self._request.set_input_tensor(idx, tensor);
        },
        py::arg("idx"),
        py::arg("tensor"));

    cls.def(
        "set_output_tensor",
        [](InferRequestWrapper& self, const ov::runtime::Tensor& tensor) {
            self._request.set_input_tensor(tensor);
        },
        py::arg("tensor"));

    cls.def("query_state", [](InferRequestWrapper& self) {
        const std::vector<ov::runtime::VariableState> states = self._request.query_state();
        return states;
    });

    cls.def_property_readonly("input_tensors", [](InferRequestWrapper& self) {
        return self._inputs;
    });

    cls.def_property_readonly("output_tensors", [](InferRequestWrapper& self) {
        return self._outputs;
    });

    cls.def_property_readonly("latency", [](InferRequestWrapper& self) {
        return self.getLatency();
    });
}
