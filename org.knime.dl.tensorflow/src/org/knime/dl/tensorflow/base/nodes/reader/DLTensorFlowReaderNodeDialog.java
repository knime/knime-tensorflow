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

import java.awt.Color;
import java.net.MalformedURLException;
import java.nio.file.InvalidPathException;
import java.util.Collection;
import java.util.Collections;

import javax.swing.JFileChooser;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.defaultnodesettings.DefaultNodeSettingsPane;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.DialogComponentStringSelection;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.defaultnodesettings.SettingsModelStringArray;
import org.knime.core.util.FileUtil;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.tensorflow.base.nodes.reader.config.DialogComponentColoredLabel;
import org.knime.dl.tensorflow.base.nodes.reader.config.DialogComponentFileOrDirChooser;
import org.knime.dl.tensorflow.base.nodes.reader.config.DialogComponentObjectSelection;
import org.knime.dl.tensorflow.base.nodes.reader.config.DialogComponentTensorSelection;
import org.knime.dl.tensorflow.savedmodel.core.DLTensorFlowMetaGraphDef;
import org.knime.dl.tensorflow.savedmodel.core.DLTensorFlowSavedModel;

/**
 * Dialog for the TensorFlow SavedModel Reader node.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowReaderNodeDialog extends DefaultNodeSettingsPane {

	private static final Collection<String> EMPTY_STRING_COLLECTION = Collections.singleton("                        ");

	private static final Collection<String[]> EMPTY_STRING_ARRAY_COLLECTION = Collections.singleton(new String[] {});

	private static final String FILE_HISTORY_ID = "org.knime.dl.tensorflow.base.nodes.reader";

	private final SettingsModelString m_smFilePath = DLTensorFlowReaderNodeModel.createFilePathSettingsModel();

	private final SettingsModelBoolean m_smCopyNetwork = DLTensorFlowReaderNodeModel.createCopyNetworkSettingsModel();

	private final SettingsModelStringArray m_smTags = DLTensorFlowReaderNodeModel.createTagsSettingsModel();

	private final SettingsModelString m_smSignature = DLTensorFlowReaderNodeModel.createSignatureSettingsModel();

	private final SettingsModelBoolean m_smAdvanced = DLTensorFlowReaderNodeModel.createAdvancedSettingsModel();

	private final SettingsModelStringArray m_smInputs = DLTensorFlowReaderNodeModel.createInputsSettingsModel();

	private final SettingsModelStringArray m_smOutputs = DLTensorFlowReaderNodeModel.createOutputsSettingsModel();

	private final DialogComponentFileOrDirChooser m_dcFiles;

	private final DialogComponentBoolean m_dcCopyNetwork;

	private final DialogComponentObjectSelection<SettingsModelStringArray, String[]> m_dcTags;

	private final DialogComponentStringSelection m_dcSignature;

	private final DialogComponentColoredLabel m_dcErrorLabel;

	private final DialogComponentBoolean m_dcAdvanced;

	private final DialogComponentTensorSelection m_dcInputs;

	private final DialogComponentTensorSelection m_dcOutputs;

	private DLTensorFlowSavedModel m_savedModel;

	private boolean m_errorReading = false;

	private String m_previousFilePath;

	/**
	 * Creates a new dialog for the TensorFlow Network Reader settings.
	 */
	public DLTensorFlowReaderNodeDialog() {
		super();

		// Create the dialog components
		m_dcFiles = new DialogComponentFileOrDirChooser(m_smFilePath, FILE_HISTORY_ID, JFileChooser.OPEN_DIALOG, null,
				".zip");
		m_dcCopyNetwork = new DialogComponentBoolean(m_smCopyNetwork,
				"Copy deep learning network into KNIME workflow?");
		m_dcTags = new DialogComponentObjectSelection<>(m_smTags, t -> String.join(" ,", t),
				(t, sm) -> sm.setStringArrayValue(t), sm -> sm.getStringArrayValue(), "Tags");
		m_dcSignature = new DialogComponentStringSelection(m_smSignature, "Signature", EMPTY_STRING_COLLECTION);
		m_dcErrorLabel = new DialogComponentColoredLabel("", Color.RED);
		m_dcAdvanced = new DialogComponentBoolean(m_smAdvanced, "Use advanced settings");
		m_dcInputs = new DialogComponentTensorSelection(m_smInputs, "Inputs", Collections.emptySet(),
				t -> DLTensorFlowReaderNodeModel.getIdentifier(t));
		m_dcOutputs = new DialogComponentTensorSelection(m_smOutputs, "Outputs", Collections.emptySet(),
				t -> DLTensorFlowReaderNodeModel.getIdentifier(t));

		// Add the dialog components
		createNewGroup("Input Location");
		addDialogComponent(m_dcFiles);
		addDialogComponent(m_dcTags);
		addDialogComponent(m_dcSignature);
		addDialogComponent(m_dcCopyNetwork);
		addDialogComponent(m_dcErrorLabel);
		closeCurrentGroup();

		createNewTab("Advanced Settings");
		addDialogComponent(m_dcAdvanced);
		addDialogComponent(m_dcInputs);
		addDialogComponent(m_dcOutputs);

		// Add change listeners
		m_smFilePath.addChangeListener(e -> {
			readSavedModel();
			updateTags();
			updateAdvanced();
		});
		m_smTags.addChangeListener(e -> updateSignatures());
		m_smAdvanced.addChangeListener(e -> updateAdvanced());

		// Switch advanced on and off to update which components are enabled
		m_smAdvanced.setBooleanValue(!m_smAdvanced.getBooleanValue());
		m_smAdvanced.setBooleanValue(!m_smAdvanced.getBooleanValue());
	}

	private void updateAdvanced() {
		final boolean advanced = m_smAdvanced.getBooleanValue();
		m_smSignature.setEnabled(!advanced);
		m_smInputs.setEnabled(advanced);
		m_smOutputs.setEnabled(advanced);
	}

	private void readSavedModel() {
		m_dcErrorLabel.setText("");
		m_errorReading = false;
		try {
			final String filePath = m_smFilePath.getStringValue();
			if (!filePath.equals(m_previousFilePath)) {
				m_savedModel = new DLTensorFlowSavedModel(FileUtil.toURL(filePath));
			}
			return;
		} catch (final DLInvalidSourceException e) {
			m_errorReading = true;
			m_dcErrorLabel.setText(e.getMessage());
		} catch (InvalidPathException | MalformedURLException e) {
			m_errorReading = true;
			m_dcErrorLabel.setText("The filepath is not valid.");
		}
		m_savedModel = null;
	}

	/**
	 * Updates the tags shown for selection in {@link #m_dcTags}. If {@link #m_savedModel} is <code>null</code> the list
	 * is set to {@link #EMPTY_STRING_COLLECTION}.
	 */
	private void updateTags() {
		if (m_savedModel != null) {
			Collection<String[]> newTagList = m_savedModel.getContainedTags();
			if (newTagList.isEmpty()) {
				newTagList = EMPTY_STRING_ARRAY_COLLECTION;
				m_dcErrorLabel.setText("The SavedModel doesn't contain tags.");
			}
			m_dcTags.replaceListItems(newTagList, null);
		} else if (m_errorReading){
			m_dcTags.replaceListItems(EMPTY_STRING_ARRAY_COLLECTION, null);
		}
	}

	/**
	 * Updates the signatures shown for selection in {@link #m_dcSignature} and the advanced signature selection.
	 */
	private void updateSignatures() {
		final String[] tags = m_smTags.getStringArrayValue();

		// Check if we can't find the signatures
		if (m_savedModel == null || tags.length < 1) {
			m_dcSignature.replaceListItems(EMPTY_STRING_COLLECTION, null);
			m_dcInputs.setTensorOptions(Collections.emptySet());
			m_dcOutputs.setTensorOptions(Collections.emptySet());
			return;
		}

		// Get the available signatures
		final DLTensorFlowMetaGraphDef metaGraphDefs = m_savedModel.getMetaGraphDefs(tags);
		final Collection<String> newSignatureList = metaGraphDefs.getSignatureDefsStrings();

		if (newSignatureList.isEmpty()) {
			// If there are no signatures activate the advanced settings
			m_smAdvanced.setBooleanValue(true);
			m_dcErrorLabel.setText(
					"The SavedModel doesn't contain signatures with the selected tag. " + "Use the advanced settings.");
			m_dcSignature.replaceListItems(EMPTY_STRING_COLLECTION, null);
		} else {
			// Else set the signatures
			m_dcSignature.replaceListItems(newSignatureList, null);
		}

		// Get the available tensors for the advanced settings
		m_dcInputs.setTensorOptions(metaGraphDefs.getPossibleInputTensors());
		m_dcOutputs.setTensorOptions(metaGraphDefs.getPossibleOutputTensors());
	}

	@Override
	public void saveAdditionalSettingsTo(final NodeSettingsWO settings) throws InvalidSettingsException {
		validateSelection();
	}

	private void validateSelection() throws InvalidSettingsException {
		if (m_savedModel == null) {
			throw new InvalidSettingsException("The path doesn't point to a valid SavedModel.");
		}
		if (m_smTags.getStringArrayValue() == null || m_smTags.getStringArrayValue().length < 1) {
			throw new InvalidSettingsException("No tags are selected.");
		}
		if (m_smSignature.getStringValue() == null || m_smSignature.getStringValue().isEmpty()) {
			throw new InvalidSettingsException("No signature is selected.");
		}
	}
}
