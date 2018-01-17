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
import org.knime.dl.tensorflow.base.portobjects.DLTensorFlowNetworkPortObject;

/**
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowReaderNodeModel extends NodeModel {

	private static final String CFG_KEY_FILE_PATH = "file_path";

	private static final String CFG_KEY_COPY_NETWORK = "copy_network";

	private static final String CFG_KEY_TAGS = "tags";

	private static final String CFG_KEY_SIGNATURES = "signatures";

	private final SettingsModelString m_filePath = createFilePathSettingsModel();

	private final SettingsModelBoolean m_copyNetwork = createCopyNetworkSettingsModel();

	private final SettingsModelStringArray m_tags = createTagsSettingsModel();

	private final SettingsModelStringArray m_signatures = createSignaturesSettingsModel();

	static SettingsModelString createFilePathSettingsModel() {
		return new SettingsModelString(CFG_KEY_FILE_PATH, "");
	}

	static SettingsModelBoolean createCopyNetworkSettingsModel() {
		return new SettingsModelBoolean(CFG_KEY_COPY_NETWORK, false);
	}

	static SettingsModelStringArray createTagsSettingsModel() {
		return new SettingsModelStringArray(CFG_KEY_TAGS, new String[] {});
	}

	static SettingsModelStringArray createSignaturesSettingsModel() {
		return new SettingsModelStringArray(CFG_KEY_SIGNATURES, new String[] {});
	}

	protected DLTensorFlowReaderNodeModel() {
		super(new PortType[] {}, new PortType[] { DLTensorFlowNetworkPortObject.TYPE });
	}

	@Override
	protected PortObjectSpec[] configure(PortObjectSpec[] inSpecs) throws InvalidSettingsException {
		// TODO read the SavedModel, create a DLTensorFlowSavedModelNetworkSpec
		// for it and create a DLTensorFlowNetworkPortObjectSpec with this
		// NetworkSpec
		return new PortObjectSpec[] { null };
	}

	@Override
	protected PortObject[] execute(PortObject[] inObjects, ExecutionContext exec) throws Exception {
		// TODO create a DLTensorFlowSavedModelNetwork and create a
		// DLTensorFlowNetworkPortObject with this network
		return new PortObject[] { new DLTensorFlowNetworkPortObject() };
	}

	@Override
	protected void loadInternals(File nodeInternDir, ExecutionMonitor exec)
			throws IOException, CanceledExecutionException {
		// nothing to do
	}

	@Override
	protected void saveInternals(File nodeInternDir, ExecutionMonitor exec)
			throws IOException, CanceledExecutionException {
		// nothing to do
	}

	@Override
	protected void saveSettingsTo(NodeSettingsWO settings) {
		m_filePath.saveSettingsTo(settings);
		m_copyNetwork.saveSettingsTo(settings);
		m_tags.saveSettingsTo(settings);
		m_signatures.saveSettingsTo(settings);
	}

	@Override
	protected void validateSettings(NodeSettingsRO settings) throws InvalidSettingsException {
		m_filePath.validateSettings(settings);
		m_copyNetwork.validateSettings(settings);
		m_tags.validateSettings(settings);
		m_signatures.validateSettings(settings);
	}

	@Override
	protected void loadValidatedSettingsFrom(NodeSettingsRO settings) throws InvalidSettingsException {
		m_filePath.loadSettingsFrom(settings);
		m_copyNetwork.loadSettingsFrom(settings);
		m_tags.loadSettingsFrom(settings);
		m_signatures.loadSettingsFrom(settings);
	}

	@Override
	protected void reset() {
		// nothing to do
	}
}
