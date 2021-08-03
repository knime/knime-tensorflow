/*
 * ------------------------------------------------------------------------
 *
 *  Copyright by KNIME AG, Zurich, Switzerland
 *  Website: http://www.knime.com; Email: contact@knime.com
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License, Version 3, as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, see <http://www.gnu.org/licenses>.
 *
 *  Additional permission under GNU GPL version 3 section 7:
 *
 *  KNIME interoperates with ECLIPSE solely via ECLIPSE's plug-in APIs.
 *  Hence, KNIME and ECLIPSE are both independent programs and are not
 *  derived from each other. Should, however, the interpretation of the
 *  GNU GPL Version 3 ("License") under any applicable laws result in
 *  KNIME and ECLIPSE being a combined program, KNIME AG herewith grants
 *  you the additional permission to use and propagate KNIME together with
 *  ECLIPSE with only the license terms in place for ECLIPSE applying to
 *  ECLIPSE and the GNU GPL Version 3 applying for KNIME, provided the
 *  license terms of ECLIPSE themselves allow for the respective use and
 *  propagation of ECLIPSE together with KNIME.
 *
 *  Additional permission relating to nodes for KNIME that extend the Node
 *  Extension (and in particular that are based on subclasses of NodeModel,
 *  NodeDialog, and NodeView) and that only interoperate with KNIME through
 *  standard APIs ("Nodes"):
 *  Nodes are deemed to be separate and independent programs and to not be
 *  covered works.  Notwithstanding anything to the contrary in the
 *  License, the License does not apply to Nodes, you are not required to
 *  license Nodes under the License, and you are granted a license to
 *  prepare and propagate Nodes, in each case even if such Nodes are
 *  propagated with or for interoperation with KNIME.  The owner of a Node
 *  may freely choose the license terms applicable to such Node, including
 *  when such Node is propagated with or for interoperation with KNIME.
 * ---------------------------------------------------------------------
 *
 */
package org.knime.dl.tensorflow.core.convert.keras;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.knime.core.data.filestore.FileStore;
import org.knime.core.util.FileUtil;
import org.knime.core.util.Version;
import org.knime.dl.base.portobjects.DLNetworkPortObject;
import org.knime.dl.core.DLCancelable;
import org.knime.dl.core.DLCanceledExecutionException;
import org.knime.dl.core.DLInvalidEnvironmentException;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.core.DLMissingExtensionException;
import org.knime.dl.core.DLNetworkFileStoreLocation;
import org.knime.dl.core.DLNetworkSpec;
import org.knime.dl.keras.base.portobjects.DLKerasNetworkPortObjectBase;
import org.knime.dl.keras.core.DLKerasNetworkSpec;
import org.knime.dl.keras.tensorflow.core.DLKerasTensorFlowNetwork;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonNetworkHandle;
import org.knime.dl.python.core.DLPythonNetworkLoaderRegistry;
import org.knime.dl.python.util.DLPythonSourceCodeBuilder;
import org.knime.dl.python.util.DLPythonUtils;
import org.knime.dl.tensorflow.core.TFNetwork;
import org.knime.dl.tensorflow.core.convert.DLNetworkConversionException;
import org.knime.dl.tensorflow.core.convert.TFAbstractNetworkConverter;
import org.knime.dl.tensorflow.savedmodel.core.TFMetaGraphDef;
import org.knime.dl.tensorflow.savedmodel.core.TFSavedModel;
import org.knime.dl.tensorflow.savedmodel.core.TFSavedModelNetwork;
import org.knime.dl.tensorflow.savedmodel.core.TFSavedModelNetworkSpec;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFKerasNetworkConverter extends TFAbstractNetworkConverter<DLPythonContext, DLKerasTensorFlowNetwork> {

	private static final String SAVE_TAG = "knime";

	private static final String SIGNATURE_KEY = "serve";

	private static final Version MIN_KERAS_VERSION = new Version(2, 1, 3);

	/**
	 * Creates a new Keras to TensorFlow converter.
	 */
	public TFKerasNetworkConverter() {
		super(DLKerasTensorFlowNetwork.class, TFSavedModelNetwork.class);
	}

	@Override
	public void checkSpec(final DLNetworkSpec spec) throws DLNetworkConversionException {
		if (!(spec instanceof DLKerasNetworkSpec)) {
			throw new DLNetworkConversionException("Cannot convert networks other than Keras networks.");
		}
		final DLKerasNetworkSpec s = (DLKerasNetworkSpec) spec;
		if (s.getKerasVersion() != null && !s.getKerasVersion().isSameOrNewer(MIN_KERAS_VERSION)) {
			throw new DLNetworkConversionException(
					"Cannot convert networks which have been created with a Keras version below "
							+ MIN_KERAS_VERSION.toString() + ".");
		}
	}

    @SuppressWarnings("resource") // Kernel will be closed along with context.
    @Override
    protected DLKerasTensorFlowNetwork extractNetworkFromPortObject(final DLPythonContext context,
        final DLNetworkPortObject networkPortObject) throws DLNetworkConversionException {
        try {
            return (DLKerasTensorFlowNetwork)((DLKerasNetworkPortObjectBase)networkPortObject)
                .getNetwork(context.getKernel().getPythonCommand());
        } catch (final DLInvalidSourceException | DLInvalidEnvironmentException | IOException e) {
            throw new DLNetworkConversionException(e.getMessage(), e);
        }
    }

	@Override
    public TFNetwork convertNetworkInternal(final DLPythonContext context, final DLKerasTensorFlowNetwork network,
        final FileStore fileStore, final DLCancelable cancelable)
        throws DLNetworkConversionException, DLCanceledExecutionException {
		try {
			final File tmpFile = new File(FileUtil.createTempDir("tf"), "sm");

			// Save the keras model as a SavedModel using python
            final DLPythonNetworkHandle networkHandle = DLPythonNetworkLoaderRegistry.getInstance()
                .getNetworkLoader(network.getClass())
                .orElseThrow(
                    () -> new DLMissingExtensionException("Python back end '" + network.getClass().getCanonicalName()
                        + "' could not be found. Are you missing a KNIME Deep Learning extension?"))
                .load(network.getSource().getURI(), context, false, cancelable);

            // Export the SavedModel to a temporary directory
            final DLPythonSourceCodeBuilder b = DLPythonUtils.createSourceCodeBuilder() //
                .a("import DLPythonNetwork") //
                .n("import keras.backend as K") //
                .n("from tensorflow import saved_model") //
                .n("model = DLPythonNetwork.get_network(").as(networkHandle.getIdentifier()).a(").model") //
                .n("inp, oup = model.input, model.output") //
                .n("builder = saved_model.builder.SavedModelBuilder(r").as(tmpFile.getAbsolutePath()).a(")") //
                .n("signature = saved_model.signature_def_utils.predict_signature_def(") //
                .n().t().a("inputs = dict([ (t.name,t) for t in (inp if type(inp) is list else [inp]) ]),") //
                .n().t().a("outputs = dict([ (t.name,t) for t in (oup if type(oup) is list else [oup]) ]))") //
                .n("signature_def_map = { ").as(SIGNATURE_KEY).a(": signature }") //
                .n("builder.add_meta_graph_and_variables(K.get_session(), [").as(SAVE_TAG)
                .a("], signature_def_map=signature_def_map)") //
                .n("builder.save()");
            context.executeInKernel(b.toString(), cancelable);

			// Move the model to the filestore
			FileUtils.moveDirectory(tmpFile, fileStore.getFile());

			// Create a TFSavedModelNetwork
			final TFSavedModel savedModel = new TFSavedModel(fileStore.getFile().toURI().toURL());
			final TFMetaGraphDef metaGraphDefs = savedModel.getMetaGraphDefs(new String[] { SAVE_TAG });
			final TFSavedModelNetworkSpec specs = metaGraphDefs.createSpecs(SIGNATURE_KEY);
			return specs.create(new DLNetworkFileStoreLocation(fileStore));
		} catch (DLInvalidSourceException | DLInvalidEnvironmentException | DLMissingExtensionException
				| IOException e) {
			throw new DLNetworkConversionException("Could not convert network.", e);
		}
	}
}
