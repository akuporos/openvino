# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest

from openvino import Core, Tensor


def image_path():
    path_to_repo = os.environ["DATA_PATH"]
    path_to_img = os.path.join(path_to_repo, "validation_set", "224x224", "dog.bmp")
    return path_to_img


def model_path(is_myriad=False):
    path_to_repo = os.environ["MODELS_PATH"]
    if not is_myriad:
        test_xml = os.path.join(path_to_repo, "models", "test_model", "test_model_fp32.xml")
        test_bin = os.path.join(path_to_repo, "models", "test_model", "test_model_fp32.bin")
    else:
        test_xml = os.path.join(path_to_repo, "models", "test_model", "test_model_fp16.xml")
        test_bin = os.path.join(path_to_repo, "models", "test_model", "test_model_fp16.bin")
    return (test_xml, test_bin)


def read_image():
    import cv2
    n, c, h, w = (1, 3, 32, 32)
    image = cv2.imread(path_to_img)
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = cv2.resize(image, (h, w)) / 255
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = image.reshape((n, c, h, w))
    return image


is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)
path_to_img = image_path()

@pytest.mark.skip(reason="ProfilingInfo has to be bound")
def test_get_profiling_info(device):
    ie_core = Core()
    func = ie_core.read_model(test_net_xml, test_net_bin)
    ie_core.set_config({"PERF_COUNT": "YES"}, device)
    exec_net = ie_core.compile_model(func, device)
    img = read_image()
    request = exec_net.create_infer_request()
    request.infer({0: img})
    pc = request.get_profiling_info() # std::vector<ProfilingInfo>  Unable to convert function return value to a Python type!

    assert pc["29"]["status"] == "EXECUTED"
    assert pc["29"]["layer_type"] == "FullyConnected"
    del exec_net
    del ie_core


def test_tensor_setter(device):
    ie_core = Core()
    func = ie_core.read_model(test_net_xml, test_net_bin)
    exec_net_1 = ie_core.compile_model(network=func, device_name=device)
    exec_net_2 = ie_core.compile_model(network=func, device_name=device)

    img = read_image()

    request1 = exec_net_1.create_infer_request()
    tensor = Tensor(img)

    request1.set_tensor("data", tensor)
    t1 = request1.get_tensor("data")

    assert np.allclose(tensor.data, t1.data, atol=1e-2, rtol=1e-2)

    res = request1.infer({0: tensor})
    res_1 = np.sort(res[0].data)
    t2 = request1.get_tensor("fc_out")
    assert np.allclose(t2.data, res[0].data, atol=1e-2, rtol=1e-2)

    request = exec_net_2.create_infer_request()
    res = request.infer({"data": tensor})
    res_2 = np.sort(request.get_tensor("fc_out").data)
    assert np.allclose(res_1, res_2, atol=1e-2, rtol=1e-2)

    request.set_tensor("data", tensor)
    t3 = request.get_tensor("data")
    assert np.allclose(t3.data, t1.data, atol=1e-2, rtol=1e-2)


def test_cancel(device):
    ie_core = Core()
    func = ie_core.read_model(test_net_xml, test_net_bin)
    exec_net = ie_core.compile_model(func, device)
    img = read_image()
    request = exec_net.create_infer_request()

    def callback(e):
        raise Exception(e)

    request.set_callback(callback)
    request.start_async({0: img})
    request.cancel()
    with pytest.raises(RuntimeError) as e:
        request.wait()
    assert "[ INFER_CANCELLED ]" in str(e.value)

    request.start_async({"data": img})
    request.cancel()
    with pytest.raises(RuntimeError) as e:
        request.wait()
    assert "[ INFER_CANCELLED ]" in str(e.value)
