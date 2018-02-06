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
package org.knime.dl.tensorflow.base.nodes.reader.config;

import java.awt.Dimension;

import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.JFileChooser;
import javax.swing.border.TitledBorder;

import org.knime.core.node.FlowVariableModel;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.defaultnodesettings.DialogComponent;
import org.knime.core.node.defaultnodesettings.DialogComponentFileChooser;
import org.knime.core.node.defaultnodesettings.SettingsModelString;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.util.FilesHistoryPanel;
import org.knime.core.node.util.FilesHistoryPanel.LocationValidation;

/**
 * Dialog component for choosing a file or a directory. Very similar to {@link DialogComponentFileChooser} but allows to
 * create and dialog which allows the selection of a file and a directory.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DialogComponentFileOrDirChooser extends DialogComponent {

	private final TitledBorder m_border;

	private final FilesHistoryPanel m_filesPanel;

	/**
	 * Creates a new {@link DialogComponentFileOrDirChooser}.
	 *
	 * @param stringModel the settings model for the path
	 * @param historyID a id for saving the history
	 * @param dialogType the dialog type of the {@link FilesHistoryPanel}
	 * @param fvm a flow variable model of null
	 * @param suffixes a list of valid filename suffixes
	 */
	public DialogComponentFileOrDirChooser(final SettingsModelString stringModel, final String historyID,
			final int dialogType, final FlowVariableModel fvm, final String... suffixes) {
		super(stringModel);

		getComponentPanel().setLayout(new BoxLayout(getComponentPanel(), BoxLayout.X_AXIS));
		m_filesPanel = new FilesHistoryPanel(fvm, historyID, LocationValidation.None, suffixes);
		m_filesPanel.setSelectMode(JFileChooser.FILES_AND_DIRECTORIES);
		m_filesPanel.setDialogType(dialogType);
		m_filesPanel.addChangeListener((e) -> {
			try {
				((SettingsModelString) getModel()).setStringValue(m_filesPanel.getSelectedFile());
			} catch (final Exception ex) {
				NodeLogger.getLogger(DialogComponentFileOrDirChooser.class)
						.error("Could not store selected file or directory in settings " + ex.getMessage(), ex);
			}
		});

		m_border = BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Selected File or Directory:");
		getComponentPanel().setBorder(m_border);
		getComponentPanel().setMaximumSize(new Dimension(Integer.MAX_VALUE, 74));
		getComponentPanel().add(m_filesPanel);
		getComponentPanel().add(Box.createHorizontalGlue());

		getModel().addChangeListener((e) -> updateComponent());
		updateComponent();
	}

	@Override
	protected void updateComponent() {
		final SettingsModelString model = (SettingsModelString) getModel();
		final String newValue = model.getStringValue();
		if ((newValue == null && !m_filesPanel.getSelectedFile().isEmpty())
				|| newValue != null && !newValue.equals(m_filesPanel.getSelectedFile())) {
			m_filesPanel.setSelectedFile(newValue);
		}
		setEnabledComponents(model.isEnabled());
	}

	@Override
	protected void validateSettingsBeforeSave() throws InvalidSettingsException {
		m_filesPanel.addToHistory();
	}

	@Override
	protected void checkConfigurabilityBeforeLoad(final PortObjectSpec[] specs) throws NotConfigurableException {
		// nothing to do
	}

	@Override
	protected void setEnabledComponents(final boolean enabled) {
		m_filesPanel.setEnabled(enabled);
	}

	@Override
	public void setToolTipText(final String text) {
		m_filesPanel.setToolTipText(text);
	}

}
