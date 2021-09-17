// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/ie_offline_transformations.hpp"

#include <generate_mapping_file.hpp>
#include <moc_transformations.hpp>
#include <pot_transformations.hpp>
#include <pruning.hpp>

// #include "ie_api_impl.hpp"
#include "openvino/pass/low_latency.hpp"
#include "openvino/pass/manager.hpp"

namespace py = pybind11;

void regmodule_offline_transformations(py::module m) {
    py::module m_offline_transformations = m.def_submodule("offline_transformations", "Offline transformations module");

    m_offline_transformations.def(
        "ApplyMOCTransformations",
        [](std::shared_ptr<ngraph::Function> function, bool cf) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::MOCTransformations>(cf);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("cf"));

    m_offline_transformations.def(
        "ApplyPOTTransformations",
        [](std::shared_ptr<ngraph::Function> function, std::string device) {
            ov::pass::Manager manager;
            manager.register_pass<ngraph::pass::POTTransformations>(std::move(device));
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("device"));

    m_offline_transformations.def(
        "ApplyLowLatencyTransformation",
        [](std::shared_ptr<ngraph::Function> function, bool use_const_initializer) {
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::LowLatency2>(use_const_initializer);
            manager.run_passes(function);
        },
        py::arg("function"),
        py::arg("use_const_initializer"));

    // m_offline_transformations.def("ApplyPruningTransformation", [](InferenceEnginePython::IENetwork network) {
    //     ov::pass::Manager manager;
    //     manager.register_pass<ngraph::pass::Pruning>();
    //     manager.run_passes(network.actual->getFunction());
    // });

    // m_offline_transformations.def("GenerateMappingFile",
    //                               [](InferenceEnginePython::IENetwork network, std::string path, bool extract_names)
    //                               {
    //                                   ov::pass::Manager manager;
    //                                   manager.register_pass<ngraph::pass::GenerateMappingFile>(path, extract_names);
    //                                   manager.run_passes(network.actual->getFunction());
    //                               });
}
