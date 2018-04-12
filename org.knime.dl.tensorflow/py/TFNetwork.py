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

import tensorflow as tf

from DLPythonNetwork import DLPythonNetwork
from DLPythonNetwork import DLPythonNetworkReader
from DLPythonNetwork import DLPythonNetworkSpec
from DLPythonNetwork import DLPythonTensorSpec

from TFModel import TFModel


class TFNetworkReader(DLPythonNetworkReader):

    def read(self, path, **kwargs):
        # TODO tags as parameter?
        tags = [ "SERVE" ]
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            tf.saved_model_loader.load(sess, tags, path)
            # TODO get the signature
        raise TFNetwork(TFModel(graph, None, tags))


class TFNetwork(DLPythonNetwork):

    def __init__(self, model):
        super().__init__(model)

    @property
    def spec(self):
        if self._spec is None:
            model = self._model

            inp_specs = [ self._tensor_spec(t, n) for n, t in model.inputs.items() ]
            hid_specs = [] # TODO hidden layer spec
            oup_specs = [ self._tensor_spec(t, n) for n, t in model.outputs.items() ]

            self._spec = TFNetworkSpec(inp_specs, hid_specs, oup_specs, model.tags)
        return self._spec

    def execute(self, in_data, batch_size):
        # TODO implement
        raise NotImplementedError()

    def save(self, path):
        model = self._model
        model.save(path)

    def _tensor_spec(self, tf_tensor, name=None):
        id = tf_tensor.name
        if name is None:
            name = tf_tensor.name
        tf_shape = tf_tensor.get_shape().as_list()
        if len(tf_shape) < 2:
            tf_shape.append(1)
        batch_size = tf_shape[0]
        shape = tf_shape[1:]
        element_type = tf_tensor.dtype.name
        try:
            data_format = tf_tensor.op.get_attr('data_format')
            dimension_order = 'TCDHW' if data_format.startswith("NC") else 'TDHWC'
        except ValueError:
            # Default to TDHWC TODO is this a good idea?
            dimension_order = 'TDHWC'
        return DLPythonTensorSpec(id, name, batch_size, shape, element_type, dimension_order)


class TFNetworkSpec(DLPythonNetworkSpec):

    def __init__(self, input_specs, intermediate_output_specs, output_specs, tags):
        super().__init__(input_specs, intermediate_output_specs, output_specs)
        self._tags = tags
        # TODO enforce the training config attribute in the super class?
        self.training_config = None

    @property
    def network_type(self):
        from TFNetworkType import instance as TensorFlow
        return TensorFlow()
