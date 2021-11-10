# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import copy
from typing import List, Union

from openvino.pyopenvino import InferRequest
from openvino.pyopenvino import ExecutableNetwork
from openvino.pyopenvino import Tensor
from openvino.utils.types import get_dtype


def normalize_inputs(py_dict: dict, py_types: dict) -> dict:
    """Normalize a dictionary of inputs to Tensors."""
    for k, v in py_dict.items():
        if isinstance(k, int):
            ov_type = list(py_types.values())[k]
        elif isinstance(k, str):
            ov_type = py_types[k]
        else:
            raise TypeError("Incompatible key type! {}".format(k))
        py_dict[k] = v if isinstance(v, Tensor) else Tensor(np.array(v, get_dtype(ov_type)))
    return py_dict


def get_input_types(obj: Union[InferRequest, ExecutableNetwork]) -> dict:
    """Get all precisions from object inputs."""
    return {i.get_node().get_friendly_name() : i.get_node().get_element_type() for i in obj.inputs}


def infer(request: InferRequest, inputs: dict = {}) -> List[np.ndarray]:
    """Infer wrapper for InferRequest."""
    res = request._infer(inputs=normalize_inputs(inputs, get_input_types(request)))
    # Required to return list since np.ndarray forces all of tensors data to match in
    # dimensions. This results in errors when running ops like variadic split.
    return [copy.deepcopy(tensor.data) for tensor in res]


def infer_new_request(exec_net: ExecutableNetwork, inputs: dict = {}) -> List[np.ndarray]:
    """Infer wrapper for ExecutableNetwork."""
    res = exec_net._infer_new_request(inputs=normalize_inputs(inputs, get_input_types(exec_net)))
    # Required to return list since np.ndarray forces all of tensors data to match in
    # dimensions. This results in errors when running ops like variadic split.
    return [copy.deepcopy(tensor.data) for tensor in res]


def start_async(request: InferRequest, inputs: dict = {}) -> None:  # type: ignore
    """Asynchronous infer wrapper for InferRequest."""
    request._start_async(inputs=normalize_inputs(inputs, get_input_types(request)))


def tensor_from_file(path: str) -> Tensor:
    """The data will be read with dtype of unit8"""
    return Tensor(np.fromfile(path, dtype=np.uint8))
