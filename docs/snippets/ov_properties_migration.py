# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.device as device
import openvino.properties.streams as streams

from utils import get_model

def main():
    core = ov.Core()

    # ! [core_set_property]
    core.set_property(device_name="CPU", properties={props.enable_profiling: True})
    # ! [core_set_property]

    model = get_model()

    if "GPU" not in core.available_devices:
        return 0

    # ! [core_compile_model]
    compiled_model = core.compile_model(model=model, device_name="MULTI", config=
        {
            device.priorities: "GPU,CPU",
            hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
            hints.inference_precision: ov.Type.f32
        })
    # ! [core_compile_model]

    # ! [compiled_model_set_property]
    # turn CPU off for multi-device execution
    compiled_model.set_property(properties={device.priorities: "GPU"})
    # ! [compiled_model_set_property]

    # ! [core_get_rw_property]
    num_streams = core.get_property("CPU", streams.num)
    # ! [core_get_rw_property]

    # ! [core_get_ro_property]
    full_device_name = core.get_property("CPU", device.full_name)
    # ! [core_get_ro_property]

    # ! [compiled_model_get_rw_property]
    perf_mode = compiled_model.get_property(hints.performance_mode)
    # ! [compiled_model_get_rw_property]

    # ! [compiled_model_get_ro_property]
    nireq = compiled_model.get_property(props.optimal_number_of_infer_requests)
    # ! [compiled_model_get_ro_property]
