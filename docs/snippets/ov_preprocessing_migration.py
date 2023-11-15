# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from utils import get_model

#! [ov_imports]
from openvino import Core, Layout, Type
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm
#! [ov_imports]


tensor_name="input"
core = Core()
model = get_model([1,32,32,3])

#! [ov_mean_scale]
ppp = PrePostProcessor(model)
input = ppp.input(tensor_name)
# we only need to know where is C dimension
input.model().set_layout(Layout('...C'))
# specify scale and mean values, order of operations is important
input.preprocess().mean([116.78]).scale([57.21, 57.45, 57.73])
# insert preprocessing operations to the 'model'
model = ppp.build()
#! [ov_mean_scale]

model = get_model()

#! [ov_conversions]
ppp = PrePostProcessor(model)
input = ppp.input(tensor_name)
input.tensor().set_layout(Layout('NCHW')).set_element_type(Type.u8)
input.model().set_layout(Layout('NCHW'))
# layout and precision conversion is inserted automatically,
# because tensor format != model input format
model = ppp.build()
#! [ov_conversions]

#! [ov_color_space]
ppp = PrePostProcessor(model)
input = ppp.input(tensor_name)
input.tensor().set_color_format(ColorFormat.NV12_TWO_PLANES)
# add NV12 to BGR conversion
input.preprocess().convert_color(ColorFormat.BGR)
# and insert operations to the model
model = ppp.build()
#! [ov_color_space]

model = get_model([1, 3, 448, 448])

#! [ov_image_scale]
ppp = PrePostProcessor(model)
input = ppp.input("input")
# need to specify H and W dimensions in model, others are not important
input.model().set_layout(Layout('??HW'))
# scale to model shape
input.preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR, 448, 448)
# and insert operations to the model
model = ppp.build()
#! [ov_image_scale]
