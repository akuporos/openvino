'''

import onnx
import io
import openvino.runtime as ov
from openvino.frontend import FrontEndManager
from onnx.helper import make_graph, make_model, make_tensor_value_info


def create_onnx_model():
    output_stream = io.BytesIO()

    add = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["z"])
    const_tensor = onnx.helper.make_tensor("const_tensor",
                                           onnx.TensorProto.FLOAT,
                                           (2, 2),
                                           [0.5, 1, 1.5, 2.0])
    const_node = onnx.helper.make_node("Constant", [], outputs=["const_node"],
                                       value=const_tensor, name="const_node")
    mul = onnx.helper.make_node("Mul", inputs=["z", "const_node"], outputs=["out"])
    input_tensors = [
        make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 2)),
        make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 2)),
    ]
    output_tensors = [make_tensor_value_info("out", onnx.TensorProto.FLOAT, (2, 2))]
    graph = make_graph([add, const_node, mul], "graph", input_tensors, output_tensors)
    model = make_model(graph, producer_name="ONNX Frontend")
    onnx.save_model(model, output_stream)
    return output_stream

ONNX_FRONTEND_NAME = "onnx"

core = ov.Core()

model = core.read_model("/home/akupuros/openvino/model.xml")
rt_info = model.get_rt_info()
print(rt_info)
#print("values")
#print(list(rt_info.values()))

#print("cycle over val")
# for key, val in rt_info.items():
#     if key == 'config':  # it is expected to have more items
#         #print(val, type(val))
#         for j in val:
#             print(j)
# model.set_rt_info('62d7a6a6587d65534beca376', ["model_parameters", "labels", "all_labels", "asdf", "id"])
# model.set_rt_info('62d7a6a6587d65534beca377', ["model_parameters", "labels", "all_labels", "fdas", "id"])
# for id in model.get_rt_info(["model_parameters", "labels", "all_labels"]):
#     print(id)

#print(list(rt_info.values()))
#print(list(rt_info.keys()))

m = b"""<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
    </edges>
    <meta_data>
        <MO_version value="TestVersion"/>
        <Runtime_version value="TestVersion"/>
        <cli_parameters>
            <input_shape value="[1, 3, 22, 22]"/>
            <transform value=""/>
            <use_new_frontend value="False"/>
        </cli_parameters>
    </meta_data>
    <framework_meta>
        <batch value="1"/>
        <chunk_size value="16"/>
    </framework_meta>
    <quantization_parameters>
        <config>{
        'compression': {
            'algorithms': [
                {
                    'name': 'DefaultQuantization',
                    'params': {
                        'num_samples_for_tuning': 2000,
                        'preset': 'performance',
                        'stat_subset_size': 300,
                        'use_layerwise_tuning': false
                    }
                }
            ],
            'dump_intermediate_model': true,
            'target_device': 'ANY'
        },
        'engine': {
            'models': [
                {
                    'name': 'bert-small-uncased-whole-word-masking-squad-0001',
                    'launchers': [
                        {
                            'framework': 'openvino',
                            'adapter': {
                                'type': 'bert_question_answering',
                                'start_token_logits_output': 'output_s',
                                'end_token_logits_output': 'output_e'
                            },
                            'inputs': [
                                {
                                    'name': 'input_ids',
                                    'type': 'INPUT',
                                    'value': 'input_ids'
                                },
                                {
                                    'name': 'attention_mask',
                                    'type': 'INPUT',
                                    'value': 'input_mask'
                                },
                                {
                                    'name': 'token_type_ids',
                                    'type': 'INPUT',
                                    'value': 'segment_ids'
                                }
                            ],
                            'device': 'cpu'
                        }
                    ],
                    'datasets': [
                        {
                            'name': 'squad_v1_1_msl384_mql64_ds128_lowercase',
                            'annotation_conversion': {
                                'converter': 'squad',
                                'testing_file': 'PATH',
                                'max_seq_length': 384,
                                'max_query_length': 64,
                                'doc_stride': 128,
                                'lower_case': true,
                                'vocab_file': 'PATH'
                            },
                            'reader': {
                                'type': 'annotation_features_extractor',
                                'features': [
                                    'input_ids',
                                    'input_mask',
                                    'segment_ids'
                                ]
                            },
                            'postprocessing': [
                                {
                                    'type': 'extract_answers_tokens',
                                    'max_answer': 30,
                                    'n_best_size': 20
                                }
                            ],
                            'metrics': [
                                {
                                    'name': 'F1',
                                    'type': 'f1',
                                    'reference': 0.9157
                                },
                                {
                                    'name': 'EM',
                                    'type': 'exact_match',
                                    'reference': 0.8504
                                }
                            ],
                            '_command_line_mapping': {
                                'testing_file': 'PATH',
                                'vocab_file': [
                                    'PATH'
                                ]
                            }
                        }
                    ]
                }
            ],
            'stat_requests_number': null,
            'eval_requests_number': null,
            'type': 'accuracy_checker'
        }
    }</config>
        <version value="invalid version"/>
        <cli_params value="{'quantize': None, 'preset': None, 'model': None, 'weights': None, 'name': None, 'engine': None, 'ac_config': None, 'max_drop': None, 'evaluate': False, 'output_dir': 'PATH', 'direct_dump': True, 'log_level': 'INFO', 'pbar': False, 'stream_output': False, 'keep_uncompressed_weights': False, 'data_source': None}"/>
    </quantization_parameters>
</net>"""

def check_rt_info(model, serialized):
    # if serialized:
    #     threshold = "13.23"
    #     min_val = "-3.24543"
    #     max_val = "3.23422"
    #     directed = "YES"
    #     empty = ""
    #     ids = "sasd fdfdfsdf"
    #     mean = "22.3 33.11 44"
    # else:
    #     threshold = 13.23
    #     min_val = -3.24543
    #     max_val = 3.234223
    #     directed = True
    #     empty = []
    #     ids = ["sasd", "fdfdfsdf"]
    #     mean = [22.3, 33.11, 44.0]
    assert model.has_rt_info(["config", "type_of_model"]) is True
    assert model.has_rt_info(["config", "converter_type"]) is True
    assert model.has_rt_info(["config", "model_parameters", "threshold"]) is True
    assert model.has_rt_info(["config", "model_parameters", "min"]) is True
    assert model.has_rt_info(["config", "model_parameters", "max"]) is True
    assert model.has_rt_info(["config", "model_parameters", "labels", "label_tree", "type"]) is True
    assert model.has_rt_info(["config", "model_parameters", "labels", "label_tree", "directed"]) is True
    assert model.has_rt_info(["config", "model_parameters", "labels", "label_tree", "float_empty"]) is True
    assert model.has_rt_info(["config", "model_parameters", "labels", "label_tree", "nodes"]) is True
    assert model.has_rt_info(["config", "model_parameters", "labels", "label_groups", "ids"]) is True
    assert model.has_rt_info(["config", "model_parameters", "mean_values"]) is True

    # assert model.get_rt_info(["config", "type_of_model"]) == "classification"
    # assert model.get_rt_info(["config", "converter_type"]) == "classification"
    # assert model.get_rt_info(["config", "model_parameters", "threshold"]) == 13.23
    # assert model.get_rt_info(["config", "model_parameters", "min"]) == -3.24543
    # assert model.get_rt_info(["config", "model_parameters", "max"]) == 3.234223
    # assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "type"]) == "tree"
    assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "directed"]).astype(bool) == True
    # assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "float_empty"]) == []
    # assert model.get_rt_info(["config", "model_parameters", "labels", "label_tree", "nodes"]) == []
    # assert model.get_rt_info(["config", "model_parameters", "labels", "label_groups", "ids"]) == ["sasd", "fdfdfsdf"]
    # assert model.get_rt_info(["config", "model_parameters", "mean_values"]) == [22.3, 33.11, 44.0]

    rt_info = model.get_rt_info()
    assert rt_info["config"] == {"converter_type": "classification",
                                    "model_parameters": {"labels": {"label_groups": {"ids": ["sasd", "fdfdfsdf"]},
                                                        "label_tree": {"directed": True, "float_empty": [],
                                                                        "nodes": [], "type": "tree"}},
                                                        "max": 3.234223, "mean_values": [22.3, 33.11, 44.0],
                                                        "min": -3.24543, "threshold": 13.23},
                                    "type_of_model": "classification"}

    # for key, value in rt_info.items():
    #     if key == "config":
    #         for config_value in value:
    #             assert config_value in ["type_of_model", "converter_type", "model_parameters"]

    # for rt_info_val in model.get_rt_info(["config", "model_parameters", "labels", "label_tree"]):
    #     assert rt_info_val in ["float_empty", "nodes", "type", "directed"]

from openvino.runtime import Core, serialize, PartialShape, Model
import openvino.runtime.opset8 as ops

core = Core()
xml_path = "/home/akupuros/openvino/openvino/serp.xml"
bin_path = "/home/akupuros/openvino/openvino/serp.bin"

# input_shape = PartialShape([1])
# param = ops.parameter(input_shape, dtype=np.float32, name="data")
# relu1 = ops.relu(param, name="relu1")
# relu1.get_output_tensor(0).set_names({"relu_t1"})
# assert "relu_t1" in relu1.get_output_tensor(0).names
# relu2 = ops.relu(relu1, name="relu2")
# model = Model(relu2, [param], "TestFunction")
model2 = core.read_model(m)

assert model2 is not None

model2.set_rt_info("classification", ["config", "type_of_model"])
model2.set_rt_info("classification", ["config", "converter_type"])
model2.set_rt_info(13.23, ["config", "model_parameters", "threshold"])
model2.set_rt_info(-3.24543, ["config", "model_parameters", "min"])
model2.set_rt_info(3.234223, ["config", "model_parameters", "max"])
model2.set_rt_info("tree", ["config", "model_parameters", "labels", "label_tree", "type"])
model2.set_rt_info(True, ["config", "model_parameters", "labels", "label_tree", "directed"])
model2.set_rt_info([], ["config", "model_parameters", "labels", "label_tree", "float_empty"])
model2.set_rt_info([], ["config", "model_parameters", "labels", "label_tree", "nodes"])
model2.set_rt_info(["sasd", "fdfdfsdf"], ["config", "model_parameters", "labels", "label_groups", "ids"])
model2.set_rt_info([22.3, 33.11, 44.0], ["config", "model_parameters", "mean_values"])

check_rt_info(model2, False)

serialize(model2, xml_path, bin_path)

#res_model = core.read_model(model=xml_path, weights=bin_path)
from openvino.frontend import FrontEndManager
fem = FrontEndManager()
fe = fem.load_by_framework("ir")
model_from_fe = fe.load(xml_path)
res_model = fe.convert(model_from_fe)
res_model_rt_info = res_model.get_rt_info()
print(xml_path)
check_rt_info(res_model, True)
'''

from openvino.runtime import serialize, opset10, PartialShape, Dimension, Type, Shape, Model, op
p1 = opset10.parameter(dtype=Type.f32, shape=PartialShape([Dimension(1), Dimension(), Dimension(1)]))
axis = op.Constant(Type.i32, Shape([0]), [])
sq = opset10.squeeze(p1, axis)
sq.output(0).get_tensor().get_rt_info()
print("one")
sq.output(0).get_tensor().get_rt_info()['my_rt_info_key'] = 'my_rt_info_value'
print("two")
my_value = sq.output(0).get_tensor().get_rt_info()['my_rt_info_key'] # check that the value is really set -- that is
res = opset10.result(sq.output(0))
model = Model([res], [p1])

serialize(model, '/home/akupuros/openvino/openvino/my_model_with_rt_info.xml')