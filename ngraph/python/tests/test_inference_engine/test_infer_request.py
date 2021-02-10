# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import pytest

import numpy as np
import os

from openvino.inference_engine import IECore, TensorDesc, Blob


def image_path():
    path_to_repo = os.environ["DATA_PATH"]
    path_to_img = os.path.join(path_to_repo, 'validation_set', '224x224', 'dog.bmp')
    return path_to_img


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp32.bin')
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.xml')
        test_bin = os.path.join(path_to_repo, "models", "test_model", 'test_model_fp16.bin')
    return (test_xml, test_bin)

path_to_image = image_path()
test_net_xml, test_net_bin = model_path()

def read_image():
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(path_to_image)
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1))
    image = image.reshape((n, c, h, w))
    return image


def test_infer(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    n, c, h, w = net.input_info[input_blob].input_data.shape

    #exec_net = ie_core.load_network(net, device, num_requests=1)
    exec_net = ie_core.load_network(net, device)
    img = read_image()
    td = TensorDesc("FP32", [n, c, h, w], "NCHW")
    input_img_blob = Blob(td, img)
    # request = exec_net.requests[0]
    request = exec_net.create_infer_request()
    request.set_input({input_blob: input_img_blob})
    # request.infer({input_blob: input_img_blob})
    request.infer()
    #res = request.output_blobs['fc_out'].buffer
    res = request.get_output_blobs([out_blob])

    assert np.argmax(res) == 2
    del exec_net
    del ie_core
    del net