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
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.net.MalformedURLException;
import java.nio.file.InvalidPathException;
import java.util.Collection;
import java.util.Collections;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;

import org.knime.core.data.DataTableSpec;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeDialogPane;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.defaultnodesettings.DialogComponentStringSelection;
import org.knime.core.node.defaultnodesettings.SettingsModelBoolean;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.defaultnodesettings.SettingsModelStringArray;
import org.knime.core.util.FileUtil;
import org.knime.core.util.ThreadPool;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.tensorflow.base.nodes.reader.config.DialogComponentFileOrDirChooser;
import org.knime.dl.tensorflow.base.nodes.reader.config.DialogComponentObjectSelection;
import org.knime.dl.tensorflow.base.nodes.reader.config.DialogComponentTensorSelection;
import org.knime.dl.tensorflow.savedmodel.core.TFMetaGraphDef;
import org.knime.dl.tensorflow.savedmodel.core.TFSavedModel;

/**
 * Dialog for the TensorFlow SavedModel Reader node.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFReaderNodeDialog extends NodeDialogPane {

	private static final NodeLogger LOGGER = NodeLogger.getLogger(TFReaderNodeDialog.class);

	private static final Collection<String> EMPTY_STRING_COLLECTION = Collections.singleton("                        ");

	private static final Collection<String[]> EMPTY_STRING_ARRAY_COLLECTION = Collections.singleton(new String[] {});

	private static final String FILE_HISTORY_ID = "org.knime.dl.tensorflow.base.nodes.reader";

	private final ThreadPool m_threadPool = new ThreadPool(100);

	private final AtomicInteger m_lastReaderID = new AtomicInteger(0);

	private final SettingsModelString m_smFilePath = TFReaderNodeModel.createFilePathSettingsModel();

	private final SettingsModelBoolean m_smCopyNetwork = TFReaderNodeModel.createCopyNetworkSettingsModel();

	private final SettingsModelStringArray m_smTags = TFReaderNodeModel.createTagsSettingsModel();

	private final SettingsModelString m_smSignature = TFReaderNodeModel.createSignatureSettingsModel();

	private final SettingsModelBoolean m_smAdvanced = TFReaderNodeModel.createAdvancedSettingsModel();

	private final SettingsModelStringArray m_smInputs = TFReaderNodeModel.createInputsSettingsModel();

	private final SettingsModelStringArray m_smOutputs = TFReaderNodeModel.createOutputsSettingsModel();

	private final DialogComponentFileOrDirChooser m_dcFiles;

	private final DialogComponentBoolean m_dcCopyNetwork;

	private final DialogComponentObjectSelection<SettingsModelStringArray, String[]> m_dcTags;

	private final DialogComponentStringSelection m_dcSignature;

	private final JLabel m_statusLabel;

	private final DialogComponentBoolean m_dcAdvanced;

	private final DialogComponentTensorSelection m_dcInputs;

	private final DialogComponentTensorSelection m_dcOutputs;

	private TFSavedModel m_savedModel;

	private boolean m_errorReading = false;

	private Future<?> m_lastReader;

	/**
	 * Creates a new dialog for the TensorFlow Network Reader settings.
	 */
	public TFReaderNodeDialog() {
		super();

		// Create the dialog components
		m_dcFiles = new DialogComponentFileOrDirChooser(m_smFilePath, FILE_HISTORY_ID, JFileChooser.OPEN_DIALOG, null,
				".zip");
		m_dcCopyNetwork = new DialogComponentBoolean(m_smCopyNetwork,
				"Copy deep learning network into KNIME workflow?");
		// We can't change the size of this component. Therefore we use it as a reference width.
		final int componentWidth = m_dcCopyNetwork.getComponentPanel().getComponent(0).getPreferredSize().width;
		m_dcTags = new DialogComponentObjectSelection<>(m_smTags, t -> String.join(" ,", t),
				(t, sm) -> sm.setStringArrayValue(t), sm -> sm.getStringArrayValue(), "Tags");
		m_dcSignature = new DialogComponentStringSelection(m_smSignature, "Signature", EMPTY_STRING_COLLECTION);
		m_statusLabel = new JLabel();
		m_dcAdvanced = new DialogComponentBoolean(m_smAdvanced, "Use advanced settings");
		m_dcInputs = new DialogComponentTensorSelection(m_smInputs, "Inputs", Collections.emptySet(),
				t -> TFReaderNodeModel.getIdentifier(t));
		m_dcOutputs = new DialogComponentTensorSelection(m_smOutputs, "Outputs", Collections.emptySet(),
				t -> TFReaderNodeModel.getIdentifier(t));

		// Add the dialog components
		final JPanel inputPanel = new JPanel(new GridBagLayout());
		final GridBagConstraints inputPanelConstr = new GridBagConstraints();
		inputPanelConstr.gridx = 0;
		inputPanelConstr.gridy = 0;
		inputPanelConstr.weightx = 1;
		inputPanelConstr.weighty = 0;
		inputPanelConstr.insets = new Insets(4, 4, 4, 4);
		inputPanelConstr.anchor = GridBagConstraints.NORTHWEST;
		inputPanelConstr.fill = GridBagConstraints.VERTICAL;

		inputPanel.add(m_dcFiles.getComponentPanel(), inputPanelConstr);
		inputPanelConstr.gridy++;
		inputPanel.add(m_dcTags.getComponentPanel(), inputPanelConstr);
		inputPanelConstr.gridy++;
		inputPanel.add(m_dcSignature.getComponentPanel(), inputPanelConstr);
		inputPanelConstr.gridy++;
		inputPanelConstr.weighty = 1;
		inputPanel.add(m_dcCopyNetwork.getComponentPanel(), inputPanelConstr);
		inputPanelConstr.gridy++;
		inputPanelConstr.anchor = GridBagConstraints.SOUTHWEST;
		inputPanel.add(m_statusLabel, inputPanelConstr);
		inputPanelConstr.gridy++;
		inputPanel.add(m_statusLabel, inputPanelConstr);
		inputPanelConstr.gridy++;

		addTab("Options", inputPanel);

		final JPanel advancedPanel = new JPanel(new GridBagLayout());
		final GridBagConstraints advancedPanelConstr = new GridBagConstraints();
		advancedPanelConstr.gridx = 0;
		advancedPanelConstr.gridy = 0;
		advancedPanelConstr.weightx = 1;
		advancedPanelConstr.weighty = 0;
		advancedPanelConstr.insets = new Insets(4, 4, 4, 4);
		advancedPanelConstr.anchor = GridBagConstraints.NORTHWEST;
		advancedPanelConstr.fill = GridBagConstraints.VERTICAL;

		advancedPanel.add(m_dcAdvanced.getComponentPanel(), advancedPanelConstr);
		advancedPanelConstr.gridy++;
		advancedPanelConstr.fill = GridBagConstraints.HORIZONTAL;
		advancedPanel.add(m_dcInputs.getComponentPanel(), advancedPanelConstr);
		advancedPanelConstr.gridy++;
		advancedPanelConstr.weighty = 1;
		advancedPanel.add(m_dcOutputs.getComponentPanel(), advancedPanelConstr);
		advancedPanelConstr.gridy++;

		addTab("Advanced Settings", advancedPanel);

		// Add change listeners
		m_smFilePath.addChangeListener(e -> readSavedModel());
		m_smTags.addChangeListener(e -> updateSignatures());
		m_smAdvanced.addChangeListener(e -> updateAdvanced());

		// Set the width of the comboboxes
		final int signatureTextWidth = new JLabel("Signature").getPreferredSize().width;
		final int comboBoxWidth = componentWidth - signatureTextWidth;
		m_dcTags.setSizeComboBox(comboBoxWidth, 24);
		m_dcTags.setSizeLabel(signatureTextWidth, 24);
		m_dcSignature.setSizeComponents(componentWidth - signatureTextWidth, 24);

		// Switch advanced on and off to update which components are enabled
		m_smAdvanced.setBooleanValue(!m_smAdvanced.getBooleanValue());
		m_smAdvanced.setBooleanValue(!m_smAdvanced.getBooleanValue());
	}

	private void updateAdvanced() {
		final boolean advanced = m_smAdvanced.getBooleanValue();
		m_smSignature.setEnabled(!advanced);
		m_smInputs.setEnabled(advanced);
		m_smOutputs.setEnabled(advanced);
		if (advanced) {
			m_dcSignature.setToolTipText(
					"Advanced settings are enabled. Configure the signature in the 'Advanced Settings' tab.");
		} else {
			m_dcSignature.setToolTipText("");
		}
	}

	private void readSavedModel() {
		final int id = m_lastReaderID.incrementAndGet();

		m_statusLabel.setForeground(Color.BLACK);
		m_statusLabel.setText("Reading SavedModel...");
		m_errorReading = false;

		// Interrupt the previous reader (may not have started jet)
		if (m_lastReader != null && !m_lastReader.isDone()) {
			m_lastReader.cancel(true);
		}
		try {
			m_lastReader = m_threadPool.submit(() -> {
				TFSavedModel savedModel = null;
				Exception exception = null;
				String errorMessage = null;

				// Try to read the saved model
				try {
					final String filePath = m_smFilePath.getStringValue();
					savedModel = new TFSavedModel(FileUtil.toURL(filePath));
				} catch (final DLInvalidSourceException e) {
					exception = e;
					errorMessage = e.getMessage();
				} catch (InvalidPathException | MalformedURLException e) {
					exception = e;
					errorMessage = "The filepath is not valid.";
				}

				// Update the UI if this is the current thread
				updateSavedModel(savedModel, exception, errorMessage, id);
			});
		} catch (InterruptedException e) {
			updateSavedModel(null, e, "Reading the SavedModel has been interrupted.", id);
		}
	}

	private synchronized void updateSavedModel(final TFSavedModel savedModel, final Exception exception,
			final String errorMessage, final int readerId) {
		if (readerId == m_lastReaderID.get()) {
			m_statusLabel.setText("");
			if (savedModel != null) {
				m_savedModel = savedModel;
			} else {
				m_savedModel = null;
				m_errorReading = true;
				LOGGER.warn(exception, exception);
				showError(errorMessage);
			}
			updateTags();
			updateAdvanced();
		}
	}

	private void showError(final String error) {
		m_statusLabel.setForeground(Color.RED);
		m_statusLabel.setText(error);
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
				showError("The SavedModel doesn't contain tags.");
			}
			m_dcTags.replaceListItems(newTagList, null);
		} else if (m_errorReading) {
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
		final TFMetaGraphDef metaGraphDefs = m_savedModel.getMetaGraphDefs(tags);
		final Collection<String> newSignatureList = metaGraphDefs.getSignatureDefsStrings();

		if (newSignatureList.isEmpty()) {
			// If there are no signatures activate the advanced settings
			m_smAdvanced.setBooleanValue(true);
			showError("The SavedModel doesn't contain signatures with the selected tag. Use the advanced settings.");
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
	protected void loadSettingsFrom(NodeSettingsRO settings, DataTableSpec[] specs) throws NotConfigurableException {
		m_dcFiles.loadSettingsFrom(settings, specs);
		m_dcCopyNetwork.loadSettingsFrom(settings, specs);
		m_dcTags.loadSettingsFrom(settings, specs);
		m_dcSignature.loadSettingsFrom(settings, specs);
		m_dcAdvanced.loadSettingsFrom(settings, specs);
		m_dcInputs.loadSettingsFrom(settings, specs);
		m_dcOutputs.loadSettingsFrom(settings, specs);
	}

	@Override
	protected void saveSettingsTo(NodeSettingsWO settings) throws InvalidSettingsException {
		m_dcFiles.saveSettingsTo(settings);
		m_dcCopyNetwork.saveSettingsTo(settings);
		m_dcTags.saveSettingsTo(settings);
		m_dcSignature.saveSettingsTo(settings);
		m_dcAdvanced.saveSettingsTo(settings);
		m_dcInputs.saveSettingsTo(settings);
		m_dcOutputs.saveSettingsTo(settings);

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
