# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.offline_transformations import ApplyMOCTransformations
#, ApplyLowLatencyTransformation, ApplyPruningTransformation, ApplyPOTTransformations

from ngraph.impl.op import Parameter
from ngraph.impl import Function, Shape, Type
from ngraph import relu


def get_test_cnnnetwork():
    element_type = Type.f32
    param = Parameter(element_type, Shape([1, 3, 22, 22]))
    op = relu(param)
    return Function([op], [param], 'test')


def test_moc_transformations():
    f = get_test_cnnnetwork()

    ApplyMOCTransformations(f, False)

    assert f != None
    assert len(f.get_ops()) == 3


# def test_low_latency_transformations():
#     net = get_test_cnnnetwork()
#     ApplyLowLatencyTransformation(net, True)

#     f = ng.function_from_cnn(net)
#     assert f != None
#     assert len(f.get_ops()) == 3


# def test_pruning_transformations():
#     net = get_test_cnnnetwork()
#     ApplyPruningTransformation(net)

#     f = ng.function_from_cnn(net)
#     assert f != None
#     assert len(f.get_ops()) == 3

# def test_pot_transformations():
#     net = get_test_cnnnetwork()
#     ApplyPOTTransformations(net, "GNA")

#     f = ng.function_from_cnn(net)
#     assert f != None
#     assert len(f.get_ops()) == 3
