# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------
#  Copyright by KNIME AG, Zurich, Switzerland
#  Website: http://www.knime.com; Email: contact@knime.com
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License, Version 3, as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses>.
#
#  Additional permission under GNU GPL version 3 section 7:
#
#  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
#  Hence, KNIME and ECLIPSE are both independent programs and are not
#  derived from each other. Should, however, the interpretation of the
#  GNU GPL Version 3 ("License") under any applicable laws result in
#  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
#  you the additional permission to use and propagate KNIME together with
#  ECLIPSE with only the license terms in place for ECLIPSE applying to
#  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
#  license terms of ECLIPSE themselves allow for the respective use and
#  propagation of ECLIPSE together with KNIME.
#
#  Additional permission relating to nodes for KNIME that extend the Node
#  Extension (and in particular that are based on subclasses of NodeModel,
#  NodeDialog, and NodeView) and that only interoperate with KNIME through
#  standard APIs ("Nodes"):
#  Nodes are deemed to be separate and independent programs and to not be
#  covered works.  Notwithstanding anything to the contrary in the
#  License, the License does not apply to Nodes, you are not required to
#  license Nodes under the License, and you are granted a license to
#  prepare and propagate Nodes, in each case even if such Nodes are
#  propagated with or for interoperation with KNIME.  The owner of a Node
#  may freely choose the license terms applicable to such Node, including
#  when such Node is propagated with or for interoperation with KNIME.
# ------------------------------------------------------------------------
'''
@author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
'''

import os
import re
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

from DLPythonDataBuffers import DLPythonBitBuffer
from DLPythonDataBuffers import DLPythonByteBuffer
from DLPythonDataBuffers import DLPythonDoubleBuffer
from DLPythonDataBuffers import DLPythonFloatBuffer
from DLPythonDataBuffers import DLPythonIntBuffer
from DLPythonDataBuffers import DLPythonLongBuffer
from DLPythonDataBuffers import DLPythonShortBuffer
from DLPythonDataBuffers import DLPythonStringBuffer
from DLPythonDataBuffers import DLPythonUnsignedByteBuffer
from DLPythonNetwork import DLPythonNetwork
from DLPythonNetwork import DLPythonNetworkReader
from DLPythonNetwork import DLPythonNetworkSpec
from DLPythonNetwork import DLPythonTensorSpec


class TF2NetworkReader(DLPythonNetworkReader):

    def read(self, path, compile=True, **kwargs):
        return TF2Network(tf.keras.models.load_model(path, compile=compile))


class TF2Network(DLPythonNetwork):

    def __init__(self, model):
        super().__init__(model)

    @property
    def spec(self):
        if self._spec is None:
            self._spec = TF2NetworkSpecExtractor(self.model).extract_spec()
        return self._spec

    def execute(self, in_data, batch_size, output_identifiers):
        if all([o.startswith('output_') for o in output_identifiers]):
            # Get a model with only the needed outputs
            model = self._get_sub_output_model(output_identifiers)
        else:
            # Get a model with the intermediate outputs
            model = self._get_hidden_output_model(output_identifiers)

        X = self._format_input(in_data, batch_size)
        Y = model.predict(X, batch_size=batch_size)
        return self._format_output(Y, output_identifiers)

    def _get_hidden_output_model(self, output_identifiers):
        """Create a model with the requested outputs. The outputs can be hidden outputs.
        The model must be a graph model.
        """
        model = self._model

        # Get the outputs
        outputs = []
        for id in output_identifiers:
            if id.startswith('output'):
                idx = int(id.split('_')[-1])
                outputs.append(model.outputs[idx])
            elif id.startswith('hidden'):
                matcher = re.match(r'^hidden/(.*)_(\d+):(\d+)$', id)
                layer_name = matcher.group(1)
                node_idx = int(matcher.group(2))
                tensor_idx = int(matcher.group(3))
                tensors = model.get_layer(layer_name).get_output_at(node_idx)
                tensors = tensors if isinstance(tensors, list) else [tensors]
                outputs.append(tensors[tensor_idx])
            else:
                raise ValueError('Unknown output requested: "{}"'.format(id))

        # Build the model with the requested outputs
        return tf.keras.Model(inputs=model.inputs, outputs=outputs)

    def _get_sub_output_model(self, output_identifiers):
        """Create a model with only the requested outputs (of the model outputs).
        The model can be any kind of model.
        """
        # Create the input tensors
        inputs = []
        for i in self.spec.input_specs:
            shape = [d if d >= 0 else None for d in i.shape]
            inputs.append(tf.keras.Input(
                shape=shape, batch_size=i.batch_size, dtype=i.element_type))

        # Apply the model
        oup = self._model(inputs)
        oup = oup if isinstance(oup, list) else [oup]

        # Get the requested output tensors
        indices = [int(o.split('_')[-1]) for o in output_identifiers]
        outputs = [oup[idx] for idx in indices]

        # Create a model with the requested output tensors
        return tf.keras.Model(inputs, outputs)

    def save(self, path):
        model = self._model
        model.save(path)

    def _format_input(self, in_data, batch_size):
        """Creates a list of numpy arrays as network input with the given input data"""
        return self._format_tensor(in_data, self.spec.input_specs, batch_size)

    def _format_output(self, Y, output_identifiers):
        if len(output_identifiers) == 1:
            Y = [Y]
        out_and_hidden_specs = self.spec.output_specs + \
            self.spec.intermediate_output_specs
        output_specs = [[s for s in out_and_hidden_specs if s.identifier == id][0]
                        for id in output_identifiers]
        output = {}
        for idx, output_spec in enumerate(output_specs):
            out = self._put_in_matching_buffer(Y[idx])
            out = pd.DataFrame({output_spec.identifier: [out]})
            output[output_spec.identifier] = out
        return output

    def _format_tensor(self, in_data, specs, batch_size):
        """Creates a list of numpy arrays with the in_data for the given specs"""
        tensors = []
        for spec in specs:
            tensor = in_data[spec.identifier].values[0][0].array
            tensor_shape = in_data[spec.identifier].values[0][1]
            tensor = tensor.reshape([batch_size] + tensor_shape)
            tensors.append(tensor)
        return tensors

    def _put_in_matching_buffer(self, y):
        if len(y.shape) < 2:
            y = y[..., None]
        t = y.dtype
        if t == np.float64:
            return DLPythonDoubleBuffer(y)
        elif t == np.float32:
            return DLPythonFloatBuffer(y)
        elif t == np.bool_:
            return DLPythonBitBuffer(y)
        elif t == np.int8:
            return DLPythonByteBuffer(y)
        elif t == np.uint8:
            return DLPythonUnsignedByteBuffer(y)
        elif t == np.int16:
            return DLPythonShortBuffer(y)
        elif t == np.int32:
            return DLPythonIntBuffer(y)
        elif t == np.int64:
            return DLPythonLongBuffer(y)
        elif t == np.object:
            return DLPythonStringBuffer(y)
        # TODO: support more types
        else:
            raise ValueError(
                'Output type of the network \'{}\' is not supported.'.format(y.dtype))


class TF2NetworkSpec(DLPythonNetworkSpec):

    def __init__(self, input_specs, intermediate_output_specs, output_specs):
        super().__init__(input_specs, intermediate_output_specs, output_specs)
        self.training_config = None

    @property
    def network_type(self):
        from TFNetworkType import instance as TensorFlow
        return TensorFlow()


class TF2NetworkSpecExtractor(object):

    def __init__(self, model):
        self._model = model

    def extract_spec(self):
        model = self._model
        dimension_order = self._determine_dimension_order()

        # Input specs
        input_specs = []
        for idx, tensor in enumerate(model.inputs):
            input_specs.append(self._create_input_tensor_specs(
                idx, tensor, dimension_order))

        # Output specs
        output_specs = []
        for idx, tensor in enumerate(model.outputs):
            output_specs.append(self._create_output_tensor_specs(
                idx, tensor, dimension_order))

        # Function for checking if a tensor is an input or output
        input_names = [t.name for t in model.inputs]
        output_names = [t.name for t in model.outputs]

        def is_input_or_output(tensor):
            return tensor.name in input_names or tensor.name in output_names

        # Hidden output specs
        hidden_specs = []
        for layer in model.layers:
            # Loop over nodes
            for node_idx in itertools.count():
                try:
                    tensors = layer.get_output_at(node_idx)
                    tensors = tensors if isinstance(
                        tensors, list) else [tensors]
                    for tensor_idx, tensor in enumerate(tensors):
                        if not is_input_or_output(tensor):
                            hidden_specs.append(self._create_hidden_tensor_specs(
                                layer, node_idx, tensor_idx, tensor, dimension_order))
                except ValueError:
                    # No node with this index
                    break

        return TF2NetworkSpec(input_specs, hidden_specs, output_specs)

    def _determine_dimension_order(self):
        """Determine the dimension order of the network.
        """
        data_format = self._determine_data_format()
        if data_format is 'channels_first':
            return 'TCDHW'
        else:
            return 'TDHWC'

    def _determine_data_format(self):
        """Determine the data format of the network using the 'data_format'
        field of the layers of the network. Either 'channels_first' or
        'channels_last'. If the data format conflicts a ValueError is raised.
        """
        data_formats = []
        for layer in self._model.layers:
            if hasattr(layer, 'data_format'):
                data_formats.append(layer.data_format)
        if len(data_formats) == 0:
            # use data format specified in keras config
            return tf.keras.backend.image_data_format()
        elif len(set(data_formats)) > 1:
            raise ValueError("The network contains conflicting data_formats.")
        else:
            # we checked that data_formats is not empty and all data_formats are the same
            return data_formats[0]

    def _create_input_tensor_specs(self, idx, tensor, dimension_order):
        """Create tensor specs with the id 'input_<idx>' and the
        name 'input_<idx>/<tensor.name>'
        """
        id = 'input_' + str(idx)
        name = id + '/' + tensor.name
        return self._create_tensor_specs(id, name, tensor, dimension_order)

    def _create_output_tensor_specs(self, idx, tensor, dimension_order):
        """Create tensor specs with the id 'output_<idx>' and the
        name 'output_<idx>/<tensor.name>'
        """
        id = 'output_' + str(idx)
        name = id + '/' + tensor.name
        return self._create_tensor_specs(id, name, tensor, dimension_order)

    def _create_hidden_tensor_specs(self, layer, node_idx, tensor_idx, tensor, dimension_order):
        """Create tensor specs with the id 'hidden/<layer.name>_<node_idx>:<tensor_idx>' and the
        name 'hidden/<tensor.name>'
        """
        id = 'hidden/' + layer.name + '_' + \
            str(node_idx) + ':' + str(tensor_idx)
        name = 'hidden/' + tensor.name
        return self._create_tensor_specs(id, name, tensor, dimension_order)

    def _create_tensor_specs(self, id, name, tensor, dimension_order):
        """Create the tensor specs for the given tensor.
        """
        # Prepare the shape
        tf_shape = tensor.shape.as_list()
        if len(tf_shape) < 2:
            tf_shape.append(1)
        batch_size = tf_shape[0]
        shape = tf_shape[1:]

        element_type = tensor.dtype.name
        return DLPythonTensorSpec(id, name, batch_size, shape, element_type, dimension_order)
