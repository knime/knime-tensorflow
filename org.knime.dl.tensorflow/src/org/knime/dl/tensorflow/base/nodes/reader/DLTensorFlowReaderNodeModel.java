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
package org.knime.dl.tensorflow.base.nodes.reader;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.file.InvalidPathException;
import java.util.Arrays;
import java.util.List;

import org.knime.core.data.filestore.FileStore;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeModel;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.defaultnodesettings.SettingsModelStringArray;
import org.knime.core.node.port.PortObject;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.port.PortType;
import org.knime.core.util.FileUtil;
import org.knime.dl.base.portobjects.DLNetworkPortObject;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.tensorflow.base.portobjects.DLTensorFlowNetworkPortObject;
import org.knime.dl.tensorflow.base.portobjects.DLTensorFlowNetworkPortObjectSpec;
import org.knime.dl.tensorflow.core.DLTensorFlowNetwork;
import org.knime.dl.tensorflow.core.DLTensorFlowSavedModel;
import org.knime.dl.tensorflow.core.DLTensorFlowSavedModelNetwork;
import org.knime.dl.tensorflow.core.DLTensorFlowSavedModelNetworkSpec;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowReaderNodeModel extends NodeModel {

	private static final String CFG_KEY_FILE_PATH = "file_path";

	private static final String CFG_KEY_COPY_NETWORK = "copy_network";

	private static final String CFG_KEY_TAGS = "tags";

	private static final String CFG_KEY_SIGNATURE = "signature";

	private static final String CFG_KEY_ADVANCED = "advanced";

	private static final String CFG_KEY_INPUTS = "inputs";

	private static final String CFG_KEY_OUTPUTS = "outputs";

	private final SettingsModelString m_filePath = createFilePathSettingsModel();

	private final SettingsModelBoolean m_copyNetwork = createCopyNetworkSettingsModel();

	private final SettingsModelStringArray m_tags = createTagsSettingsModel();

	private final SettingsModelString m_signatures = createSignatureSettingsModel();

	private final SettingsModelBoolean m_advanced = createAdvancedSettingsModel();

	private final SettingsModelStringArray m_inputs = createInputsSettingsModel();

	private final SettingsModelStringArray m_outputs = createOutputsSettingsModel();

	private DLTensorFlowSavedModelNetworkSpec m_networkSpec;

	private URL m_url;

	static SettingsModelString createFilePathSettingsModel() {
		return new SettingsModelString(CFG_KEY_FILE_PATH, "");
	}

	static SettingsModelBoolean createCopyNetworkSettingsModel() {
		return new SettingsModelBoolean(CFG_KEY_COPY_NETWORK, false);
	}

	static SettingsModelStringArray createTagsSettingsModel() {
		return new SettingsModelStringArray(CFG_KEY_TAGS, new String[] {});
	}

	static SettingsModelString createSignatureSettingsModel() {
		return new SettingsModelString(CFG_KEY_SIGNATURE, "");
	}

	static SettingsModelBoolean createAdvancedSettingsModel() {
		return new SettingsModelBoolean(CFG_KEY_ADVANCED, false);
	}

	static SettingsModelStringArray createInputsSettingsModel() {
		return new SettingsModelStringArray(CFG_KEY_INPUTS, new String[0]);
	}

	static SettingsModelStringArray createOutputsSettingsModel() {
		return new SettingsModelStringArray(CFG_KEY_OUTPUTS, new String[0]);
	}

	static String getIdentifier(final DLTensorSpec t) {
		// The names in a TensorFlow graph are unique
		return t.getName();
	}

	/**
	 * Creates a new {@link NodeModel} for the TensorFlow Network Reader.
	 */
	protected DLTensorFlowReaderNodeModel() {
		super(new PortType[] {}, new PortType[] { DLTensorFlowNetworkPortObject.TYPE });
	}

	@Override
	protected PortObjectSpec[] configure(final PortObjectSpec[] inSpecs) throws InvalidSettingsException {
		try {
			m_url = FileUtil.toURL(m_filePath.getStringValue());
		} catch (InvalidPathException | MalformedURLException e) {
			throw new InvalidSettingsException("The file path is not valid.", e);
		}
		final String signature = m_signatures.getStringValue();
		try {
			final DLTensorFlowSavedModel sm = new DLTensorFlowSavedModel(m_url);
			// Create the NetworkSpec
			if (!m_advanced.getBooleanValue()) {
				m_networkSpec = sm.createSpecs(m_tags.getStringArrayValue(), signature);
			} else {
				final List<String> tags = Arrays.asList(m_tags.getStringArrayValue());
				final List<String> inputs = Arrays.asList(m_inputs.getStringArrayValue());
				final List<String> outputs = Arrays.asList(m_outputs.getStringArrayValue());
				final DLTensorSpec[] inputSpecs = sm.getPossibleInputTensors(tags).stream()
						.filter(t -> inputs.contains(getIdentifier(t))).toArray(s -> new DLTensorSpec[s]);
				final DLTensorSpec[] hiddenSpecs = new DLTensorSpec[0];
				final DLTensorSpec[] outputSpecs = sm.getPossibleOutputTensors(tags).stream()
						.filter(t -> outputs.contains(getIdentifier(t))).toArray(s -> new DLTensorSpec[s]);
				m_networkSpec = new DLTensorFlowSavedModelNetworkSpec(m_tags.getStringArrayValue(), inputSpecs,
						hiddenSpecs, outputSpecs);
			}
		} catch (final DLInvalidSourceException e) {
			throw new InvalidSettingsException("The file is not a valid SavedModel.", e);
		}
		return new PortObjectSpec[] {
				new DLTensorFlowNetworkPortObjectSpec(m_networkSpec, DLTensorFlowSavedModelNetwork.class) };

	}

	@Override
	protected PortObject[] execute(final PortObject[] inObjects, final ExecutionContext exec) throws Exception {
		if (m_url == null || m_networkSpec == null) {
			throw new RuntimeException("Please run configure first.");
		}
		// Create the network object
		final DLTensorFlowNetwork network = m_networkSpec.create(m_url);
		DLTensorFlowNetworkPortObject portObject;
		if (m_copyNetwork.getBooleanValue()) {
			final FileStore fileStore = DLNetworkPortObject.createFileStoreForSaving(null, exec);
			portObject = new DLTensorFlowNetworkPortObject(network, fileStore);
		} else {
			portObject = new DLTensorFlowNetworkPortObject(network);
		}
		return new PortObject[] { portObject };
	}

	@Override
	protected void loadInternals(final File nodeInternDir, final ExecutionMonitor exec)
			throws IOException, CanceledExecutionException {
		// nothing to do
	}

	@Override
	protected void saveInternals(final File nodeInternDir, final ExecutionMonitor exec)
			throws IOException, CanceledExecutionException {
		// nothing to do
	}

	@Override
	protected void saveSettingsTo(final NodeSettingsWO settings) {
		m_filePath.saveSettingsTo(settings);
		m_copyNetwork.saveSettingsTo(settings);
		m_tags.saveSettingsTo(settings);
		m_signatures.saveSettingsTo(settings);
		m_advanced.saveSettingsTo(settings);
		m_inputs.saveSettingsTo(settings);
		m_outputs.saveSettingsTo(settings);
	}

	@Override
	protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
		m_filePath.validateSettings(settings);
		m_copyNetwork.validateSettings(settings);
		m_tags.validateSettings(settings);
		m_signatures.validateSettings(settings);
		m_advanced.validateSettings(settings);
		m_inputs.validateSettings(settings);
		m_outputs.validateSettings(settings);
	}

	@Override
	protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
		m_filePath.loadSettingsFrom(settings);
		m_copyNetwork.loadSettingsFrom(settings);
		m_tags.loadSettingsFrom(settings);
		m_signatures.loadSettingsFrom(settings);
		m_advanced.loadSettingsFrom(settings);
		m_inputs.loadSettingsFrom(settings);
		m_outputs.loadSettingsFrom(settings);
	}

	@Override
	protected void reset() {
		m_url = null;
		m_networkSpec = null;
	}
}
