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

import org.knime.dl.core.DLInvalidEnvironmentException;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.python.core.DLPythonAbstractCommands;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonNetworkHandle;
import org.knime.dl.python.core.DLPythonNumPyTypeMap;
import org.knime.dl.python.core.DLPythonTensorSpecTableCreatorFactory;
import org.knime.dl.python.util.DLPythonSourceCodeBuilder;
import org.knime.dl.python.util.DLPythonUtils;
import org.knime.dl.util.DLUtils;
import org.knime.python2.extensions.serializationlibrary.interfaces.Row;
import org.knime.python2.extensions.serializationlibrary.interfaces.TableCreator;
import org.knime.python2.extensions.serializationlibrary.interfaces.TableSpec;
import org.knime.python2.kernel.PythonKernel;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public final class TFPythonCommands extends DLPythonAbstractCommands {

	/**
	 * Creates new TensorFlow python commands without a context.
	 */
	public TFPythonCommands() {
	}

	/**
	 * Creates new TensorFlow python commands with a context.
	 *
	 * @param context the python context
	 */
	public TFPythonCommands(final DLPythonContext context) {
		super(context);
	}

	@Override
	public TFSavedModelNetworkSpec extractNetworkSpec(final DLPythonNetworkHandle network)
			throws DLInvalidEnvironmentException, IOException {
		getContext().executeInKernel(getExtractNetworkSpecsCode(network));
		final PythonKernel kernel = getContext().getKernel();
		final DLTensorSpec[] inputSpecs = (DLTensorSpec[]) kernel
				.getData(INPUT_SPECS_NAME, new DLPythonTensorSpecTableCreatorFactory(DLPythonNumPyTypeMap.INSTANCE))
				.getTable();
		final DLTensorSpec[] hiddenOutputSpecs = new DLTensorSpec[0];
//		DLTensorSpec[] hiddenOutputSpecs;
//		try {
//			hiddenOutputSpecs = (DLTensorSpec[]) kernel.getData(HIDDEN_OUTPUT_SPECS_NAME,
//					new DLPythonTensorSpecTableCreatorFactory(DLPythonNumPyTypeMap.INSTANCE)).getTable();
//		} catch (final IllegalStateException e) {
//			// We didn't get the hidden specs
//			hiddenOutputSpecs = new DLTensorSpec[0];
//		}
		final DLTensorSpec[] outputSpecs = (DLTensorSpec[]) kernel
				.getData(OUTPUT_SPECS_NAME, new DLPythonTensorSpecTableCreatorFactory(DLPythonNumPyTypeMap.INSTANCE))
				.getTable();

		getContext().executeInKernel(getExtractTagsCode(network));
		final String[] tags = (String[]) kernel.getData("tags", (spec, tableSize) -> new TableCreator<String[]>() {

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
		}).getTable();

		return new TFSavedModelNetworkSpec(tags, inputSpecs, hiddenOutputSpecs, outputSpecs);
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

	private String getExtractTagsCode(final DLPythonNetworkHandle network) {
		return "import pandas as pd\n" + //
				"global tags\n" + //
				"tags = pd.DataFrame({ 'tags':" + network.getIdentifier() + ".tags })";
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
