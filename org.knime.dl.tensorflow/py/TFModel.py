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
import shutil
import tempfile

import tensorflow as tf
# TODO better documentation


class TFModel(object):
    """ Object which holds all information for a TensorFlow model.
    In contrast to an SavedModel this object only supports one signature.
    """

    def __init__(self, inputs, outputs, graph=None, session=None, tags=['SAVE'],
                 method_name='PREDICT', signature_key='predict', save=True):

        if graph is None and session is None:
            # Both None is forbidden
            raise ValueError('Either a graph or a session must be given.')
        elif graph is None:
            # We use the graph from the session
            graph = session.graph
        elif session is not None:
            # Both are not None. Check that they fit
            if session.graph is not graph:
                raise ValueError('The given session is not on the given graph.')

        if not isinstance(inputs, dict):
            raise ValueError('The inputs must be a dictionary.')
        if not isinstance(outputs, dict):
            raise ValueError('The outputs must be a dictionary.')

        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
        self.session = session
        self.tags = tags
        self._method_name = method_name
        self._signature_key = signature_key

        # Save the variables to a temporary directory
        if save:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._sm_path = os.path.join(self._tmpdir.name, 'sm')
            self._save_saved_model(self._sm_path)
        else:
            raise ValueError('A session must be given if save is False')

    def save(self, path):
        if hasattr(self, '_sm_path'):
            # Move the SavedModel to the filestore
            shutil.move(self._sm_path, path)
            # Cleanup the temporary directory
            self._tmpdir.cleanup()
        else:
            self._save_saved_model(path)
            self.session.close()

    def _save_saved_model(self, path):
        if self.session is not None:
            with self.session.as_default() as sess:
                with sess.graph.as_default():
                    self._save_saved_model_with_session(sess, path)
        else:
            with tf.Session(graph=self.graph) as sess:
                # Initialize the variables
                sess.run(tf.global_variables_initializer())
                self._save_saved_model_with_session(sess, path)

    def _save_saved_model_with_session(self, session, path):
        builder = tf.saved_model.builder.SavedModelBuilder(path)
        inps = self.inputs
        oups = self.outputs
        tensor_info = tf.saved_model.utils.build_tensor_info
        sig_inps = { k: tensor_info(t) for k, t in inps.items() }
        sig_oups = { k: tensor_info(t) for k, t in oups.items() }
        sig = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=sig_inps,
            outputs=sig_oups,
            method_name=self._method_name)
        sigs = { self._signature_key: sig }
        builder.add_meta_graph_and_variables(session,
                                             self.tags,
                                             signature_def_map=sigs)
        builder.save()
