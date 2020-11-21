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
package org.knime.dl.tensorflow.savedmodel.core;

import java.io.File;
import java.io.IOException;

import org.knime.core.util.Version;
import org.knime.dl.core.DLCancelable;
import org.knime.dl.core.DLCanceledExecutionException;
import org.knime.dl.core.DLInvalidEnvironmentException;
import org.knime.dl.core.DLNetworkInputProvider;
import org.knime.dl.core.DLTensor;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.data.DLWritableBuffer;
import org.knime.dl.core.training.DLTrainingMonitor;
import org.knime.dl.python.core.DLPythonAbstractCommands;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonNetworkHandle;
import org.knime.dl.python.core.SingleValueTableCreator;
import org.knime.dl.python.core.training.DLPythonTrainingStatus;
import org.knime.dl.python.util.DLPythonSourceCodeBuilder;
import org.knime.dl.python.util.DLPythonUtils;
import org.knime.dl.tensorflow.core.TFUtil;
import org.knime.dl.util.DLThrowingLambdas.DLThrowingBiFunction;
import org.knime.dl.util.DLUtils;
import org.knime.python2.extensions.serializationlibrary.interfaces.Cell;
import org.knime.python2.extensions.serializationlibrary.interfaces.Row;
import org.knime.python2.extensions.serializationlibrary.interfaces.TableChunker;
import org.knime.python2.extensions.serializationlibrary.interfaces.TableCreator;
import org.knime.python2.extensions.serializationlibrary.interfaces.TableSpec;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public final class TFPythonCommands extends DLPythonAbstractCommands {

	private static final String TF_VERSION_NAME = "tf_version";

	/**
	 * Creates new TensorFlow python commands with a context.
	 *
	 * @param context the python context
	 */
	public TFPythonCommands(final DLPythonContext context) {
		super(context);
	}

	@Override
	public TFSavedModelNetworkSpec extractNetworkSpec(final DLPythonNetworkHandle network,
			final DLCancelable cancelable)
			throws DLInvalidEnvironmentException, IOException, DLCanceledExecutionException {
		getContext(cancelable).executeInKernel(getExtractNetworkSpecsCode(network), cancelable);
		final DLTensorSpec[] inputSpecs = extractTensorSpec(INPUT_SPECS_NAME, cancelable);
		final DLTensorSpec[] hiddenOutputSpecs = new DLTensorSpec[0];
		// DLTensorSpec[] hiddenOutputSpecs;
		// try {
		// hiddenOutputSpecs = extractTensorSpec(HIDDEN_OUTPUT_SPECS_NAME, cancelable);
		// } catch (final IllegalStateException e) {
		// // We didn't get the hidden specs
		// hiddenOutputSpecs = new DLTensorSpec[0];
		// }
		final DLTensorSpec[] outputSpecs = extractTensorSpec(OUTPUT_SPECS_NAME, cancelable);

		getContext(cancelable).executeInKernel(getExtractTagsCode(network), cancelable);
		final String[] tags = (String[]) getContext(cancelable)
				.getDataFromKernel("tags", (spec, tableSize) -> new TableCreator<String[]>() {

					private final String[] m_tags = new String[tableSize];
					private int m_nextIdx = 0;

					@Override
					public void addRow(final Row row) {
						m_tags[m_nextIdx++] = row.getCell(0).getStringValue();
					}

					@Override
					public TableSpec getTableSpec() {
						return spec;
					}

					@Override
					public String[] getTable() {
						return m_tags;
					}
				}, cancelable).getTable();

		// Get the version numbers
		final Version pythonVersion = getPythonVersion(cancelable);
		final Version tfVersion = getTensorFlowVersion(cancelable);
		TFUtil.checkTFVersion(tfVersion);

		return new TFSavedModelNetworkSpec(pythonVersion, tfVersion, tags, inputSpecs, hiddenOutputSpecs, outputSpecs);
	}

	@Override
	protected String getSetupEnvironmentCode() {
		return "";
	}

	@Override
	protected File getInstallationTestFile() throws IOException {
		return DLUtils.Files.getFileFromSameBundle(this, "py/TFNetworkTester.py");
	}

	@Override
	protected String getSetupBackendCode() {
		return "";
	}

	@Override
	protected TFPythonNetworkReaderCommands getNetworkReaderCommands() {
		return new TFPythonNetworkReaderCommands();
	}

	@Override
	protected DLPythonNetworkTrainingTaskHandler createNetworkTrainingTaskHandler(final DLPythonContext context,
			final DLTrainingMonitor<? extends DLPythonTrainingStatus> monitor,
			final DLNetworkInputProvider trainingInputProvider, final DLNetworkInputProvider validationInputProvider,
			final DLThrowingBiFunction<DLTensorId, DLTensor<? extends DLWritableBuffer>, TableChunker, IOException> singleTensorTableChunkerCreator) {
		throw new UnsupportedOperationException("TensorFlow does not support training networks, yet.");
	}

	private String getExtractTagsCode(final DLPythonNetworkHandle network) {
		// NB: We need to write sequences into a new list because pandas<0.21 can't handle all types of sequence
		return "import pandas as pd\n" + //
				"import collections\n" + //
				"global tags\n" + //
				"tags_val = " + network.getIdentifier() + ".tags\n" + //
				"if isinstance(tags_val, collections.Sequence) and not isinstance(tags_val, str):\n" + //
				"\ttags_val = [ t for t in tags_val ]\n" + //
				"else:\n" + //
				"\ttags_val = [ tags_val ]\n" + //
				"tags = pd.DataFrame({ 'tags': tags_val })";
	}

	/**
	 * @param cancelable to check if the execution has been canceled
	 * @return the TensorFlow version
	 * @throws DLCanceledExecutionException if the execution has been canceled
	 * @throws DLInvalidEnvironmentException if failed to properly setup the Python context
	 * @throws IOException if getting the data from python failed
	 */
	protected Version getTensorFlowVersion(final DLCancelable cancelable)
			throws DLCanceledExecutionException, DLInvalidEnvironmentException, IOException {
		final DLPythonSourceCodeBuilder b = DLPythonUtils.createSourceCodeBuilder() //
				.a("import tensorflow as tf") //
				.n("import pandas as pd") //
				.n("global ").a(TF_VERSION_NAME) //
				.n(TF_VERSION_NAME).a(" = pd.DataFrame([tf.__version__])"); //
		getContext(cancelable).executeInKernel(b.toString(), cancelable);
		final String kerasVersion = (String) getContext(cancelable).getDataFromKernel(TF_VERSION_NAME,
				(s, ts) -> new SingleValueTableCreator<>(s, Cell::getStringValue), cancelable).getTable();
		return new Version(kerasVersion);
	}

	private static class TFPythonNetworkReaderCommands extends DLPythonAbstractNetworkReaderCommands {

		private TFPythonNetworkReaderCommands() {
			super("from TFNetwork import TFNetworkReader", "TFNetworkReader()");
		}

		@Override
		public String read(final String path, final boolean loadTrainingConfig) {
			final DLPythonSourceCodeBuilder b = DLPythonUtils.createSourceCodeBuilder() //
					.a("read(").asr(path).a(")");
			return b.toString();
		}
	}
}
