# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest

from openvino.inference_engine import IECore, Blob, TensorDesc, PreProcessInfo, MeanVariant, ResizeAlgorithm
from ..conftest import model_path, image_path, read_image


is_myriad = os.environ.get("TEST_DEVICE") == "MYRIAD"
test_net_xml, test_net_bin = model_path(is_myriad)
path_to_img = image_path()

def test_get_perf_counts(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    ie_core.set_config({"PERF_COUNT": "YES"}, device)
    exec_net = ie_core.load_network(net, device)
    img = read_image()
    request = exec_net.create_infer_request()
    td = TensorDesc("FP32", [1, 3, 32, 32], "NCHW")
    input_blob = Blob(td, img)
    request.set_input({'data': input_blob})
    request.infer()
    pc = request.get_perf_counts()
    assert pc['29']["status"] == "EXECUTED"
    assert pc['29']["layer_type"] == "FullyConnected"
    del exec_net
    del ie_core
    del net


@pytest.mark.skipif(os.environ.get("TEST_DEVICE", "CPU") != "CPU",
                    reason=f"Can't run test on device {os.environ.get('TEST_DEVICE', 'CPU')}, "
                           "Dynamic batch fully supported only on CPU")
def test_set_batch_size(device):
    ie_core = IECore()
    ie_core.set_config({"DYN_BATCH_ENABLED": "YES"}, device)
    net = ie_core.read_network(test_net_xml, test_net_bin)
    net.batch_size = 10
    data = np.ones(shape=net.input_info['data'].input_data.shape,dtype = np.float32)
    exec_net = ie_core.load_network(net, device)
    data[0] = read_image()[0]
    request = exec_net.create_infer_request()
    request.set_batch(1)
    td = TensorDesc("FP32", [10, 3, 32, 32], "NCHW")
    input_blob = Blob(td, data)
    request.set_input({'data': input_blob})
    request.infer()
    assert np.allclose(int(round(request.output_blobs['fc_out'].buffer[0][2])), 1), "Incorrect data for 1st batch"
    del exec_net
    del ie_core
    del net


def test_set_zero_batch_size(device):
    ie_core = IECore()
    ie_core.set_config({"DYN_BATCH_ENABLED": "YES"}, device)
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device)
    request = exec_net.create_infer_request()
    with pytest.raises(RuntimeError) as e:
        request.set_batch(0)
    assert "Invalid dynamic batch size 0 for this request" in str(e.value)
    del exec_net
    del ie_core
    del net


def test_set_negative_batch_size(device):
    ie_core = IECore()
    ie_core.set_config({"DYN_BATCH_ENABLED": "YES"}, device)
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(net, device)
    request = exec_net.create_infer_request()
    with pytest.raises(RuntimeError) as e:
        request.set_batch(-1)
    assert "Invalid dynamic batch size -1 for this request" in str(e.value)
    del exec_net
    del ie_core
    del net


def test_getting_preprocess(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(network=net, device_name=device)
    request = exec_net.create_infer_request()
    preprocess_info = request.preprocess_info("data")
    assert isinstance(preprocess_info, PreProcessInfo)
    assert preprocess_info.mean_variant == MeanVariant.NONE


def test_resize_algorithm_work(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net_1 = ie_core.load_network(network=net, device_name=device)
    request1 = exec_net_1.create_infer_request()

    img = read_image()

    tensor_desc = TensorDesc("FP32", [1, 3, img.shape[2], img.shape[3]], "NCHW")
    img_blob1 = Blob(tensor_desc, img)
    request1.set_input({'data': img_blob1})
    request1.infer()
    res_1 = np.sort(request1.get_blob('fc_out').buffer)

    net.input_info['data'].preprocess_info.resize_algorithm = ResizeAlgorithm.RESIZE_BILINEAR

    exec_net_2 = ie_core.load_network(net, device)

    import cv2

    image = cv2.imread(path_to_img)
    if image is None:
        raise FileNotFoundError("Input image not found")

    image = image / 255
    image = image.transpose((2, 0, 1)).astype(np.float32)
    image = np.expand_dims(image, 0)

    tensor_desc = TensorDesc("FP32", [1, 3, image.shape[2], image.shape[3]], "NCHW")
    img_blob = Blob(tensor_desc, image)
    request = exec_net_2.create_infer_request()
    assert request.preprocess_info("data").resize_algorithm == ResizeAlgorithm.RESIZE_BILINEAR
    request.set_input({'data': img_blob})
    request.infer()
    res_2 = np.sort(request.get_blob('fc_out').buffer)

    assert np.allclose(res_1, res_2, atol=1e-2, rtol=1e-2)


def test_blob_setter(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net_1 = ie_core.load_network(network=net, device_name=device)

    net.input_info['data'].layout = "NHWC"
    exec_net_2 = ie_core.load_network(network=net, device_name=device)

    img = read_image()

    request1 = exec_net_1.create_infer_request()
    tensor_desc = TensorDesc("FP32", [1, 3, img.shape[2], img.shape[3]], "NCHW")
    img_blob1 = Blob(tensor_desc, img)
    request1.set_input({'data': img_blob1})
    request1.infer()
    res_1 = np.sort(request1.get_blob('fc_out').buffer)

    img = np.transpose(img, axes=(0, 2, 3, 1)).astype(np.float32)
    tensor_desc = TensorDesc("FP32", [1, 3, 32, 32], "NHWC")
    img_blob = Blob(tensor_desc, img)
    request = exec_net_2.create_infer_request()
    request.set_blob('data', img_blob)
    request.infer()
    res_2 = np.sort(request.get_blob('fc_out').buffer)
    assert np.allclose(res_1, res_2, atol=1e-2, rtol=1e-2)


def test_blob_setter_with_preprocess(device):
    ie_core = IECore()
    net = ie_core.read_network(test_net_xml, test_net_bin)
    exec_net = ie_core.load_network(network=net, device_name=device)

    img = read_image()
    tensor_desc = TensorDesc("FP32", [1, 3, 32, 32], "NCHW")
    img_blob = Blob(tensor_desc, img)
    preprocess_info = PreProcessInfo()
    preprocess_info.mean_variant = MeanVariant.MEAN_IMAGE

    request = exec_net.create_infer_request()
    request.set_blob('data', img_blob, preprocess_info)
    pp = request.preprocess_info("data")
    assert pp.mean_variant == MeanVariant.MEAN_IMAGE
