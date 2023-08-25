# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


#! [static_shape]
import openvino as ov

core = ov.Core()
model = core.read_model("model.xml")
model.reshape([10, 20, 30, 40])
#! [static_shape]
