//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <boost/type_index.hpp>

#include <string>
#include <vector>

#include <cpp/ie_infer_request.hpp>

#include "../../../pybind11/include/pybind11/pybind11.h"
#include "pyopenvino/inference_engine/ie_executable_network.hpp"

namespace py = pybind11;

void regclass_InferRequest(py::module m)
{
    py::class_<InferenceEngine::InferRequest, std::shared_ptr<InferenceEngine::InferRequest>> cls(
        m, "InferRequest");

    cls.def("infer", &InferenceEngine::InferRequest::Infer);
    cls.def("get_blob", &InferenceEngine::InferRequest::GetBlob);
    cls.def("set_input", [](InferenceEngine::InferRequest& self, const py::dict& inputs) {
        for (auto&& input : inputs) {
            auto name = input.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::TBlob<float>>& blob = input.second.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
            self.SetBlob(name, blob);
        }
    });
    cls.def("set_output", [](InferenceEngine::InferRequest& self, const py::dict& results) {
        for (auto&& result : results) {
            auto name = result.first.cast<std::string>().c_str();
            const std::shared_ptr<InferenceEngine::TBlob<float>>& blob = result.second.cast<const std::shared_ptr<InferenceEngine::TBlob<float>>&>();
            self.SetBlob(name, blob);
        }
    });
    cls.def("get_output_blobs", [](InferenceEngine::InferRequest& self, const py::list& output_names) {
        #define STORE_TBLOB(precision)                                                                              \
        case InferenceEngine::Precision::precision: {                                                               \
            using myBlobType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::precision>::value_type;  \
            InferenceEngine::TBlob<myBlobType>& tblob = dynamic_cast<InferenceEngine::TBlob<myBlobType>&>(*blob);    \
            auto blob_ptr = tblob.buffer().template as<myBlobType*>();                                              \
            auto shape = tblob.getTensorDesc().getDims();                                                           \
            output_blobs[outname] = py::array_t<myBlobType>(shape, blob_ptr);                                       \
            break;                                                                                                  \
        }

        py::dict output_blobs;
        for (auto&& outname : output_names) {
            InferenceEngine::Blob::Ptr blob = self.GetBlob(outname.cast<std::string>().c_str());
            auto precision = blob->getTensorDesc().getPrecision();

            switch (precision) {
                STORE_TBLOB(FP32);
                STORE_TBLOB(FP16);
                STORE_TBLOB(Q78);
                STORE_TBLOB(I16);
                STORE_TBLOB(U8);
                STORE_TBLOB(I8);
                STORE_TBLOB(U16);
                STORE_TBLOB(I32);
                STORE_TBLOB(U32);
                STORE_TBLOB(U64);
                STORE_TBLOB(I64);
                default:
                    THROW_IE_EXCEPTION << "cannot locate blob for precision: " << precision;
            }
        }
        #undef STORE_TBLOB
        return output_blobs;
    });
//    cls.def_property_readonly("input_blobs", [](){
//
//    });
//    cls.def_property_readonly("output_blobs", [](InferenceEngine::InferRequest& self) {
//        py::ExecutableNetwork exe;
//        auto exe.outputs;
//    });
//    cls.def("set_batch", );
//    cls.def("get_perf_counts", );
//    cls.def("wait");
//    cls.def("set_completion_callback")
//    cls.def("async_infer",);
//    latency
//    cls.def_property_readonly("preprocess_info", [](){
//
//    });
//   set_blob

    //&InferenceEngine::InferRequest::SetOutput);
}
