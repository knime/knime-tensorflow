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
 * History
 *   May 26, 2020 (benjamin): created
 */
package org.knime.dl.tensorflow2.base.nodes.io.filehandling.tfnetwork.writer;

import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;

import javax.swing.BorderFactory;
import javax.swing.JPanel;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.defaultnodesettings.DialogComponentBoolean;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.filehandling.core.node.portobject.writer.PortObjectWriterNodeDialog;
import org.knime.python2.config.PythonCommandFlowVariableModel;

/**
 * Node dialog of the TensorFlow writer node.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
final class TF2WriterNodeDialog extends PortObjectWriterNodeDialog<TF2WriterNodeConfig> {

    private final PythonCommandFlowVariableModel m_pythonCommandModel =
        new PythonCommandFlowVariableModel(this, TF2WriterNodeModel.createPythonCommandConfig());

    private final DialogComponentBoolean m_saveOptimizerStateCheckbox;

    TF2WriterNodeDialog(final TF2WriterNodeConfig config, final String fileChooserHistoryId) {
        super(config, fileChooserHistoryId);
        m_saveOptimizerStateCheckbox =
            new DialogComponentBoolean(config.getSaveOptimizerStateModel(), "Save Optimizer State");
        addAdditionalPanel(createNetworkSettingsPanel());
    }

    private JPanel createNetworkSettingsPanel() {
        final JPanel panel = new JPanel(new GridBagLayout());
        panel.setBorder(BorderFactory.createTitledBorder(BorderFactory.createEtchedBorder(), "Network"));
        final GridBagConstraints gbc = createAndInitGBC();
        gbc.weightx = 1;
        panel.add(m_saveOptimizerStateCheckbox.getComponentPanel(), gbc);
        return panel;
    }

    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) throws InvalidSettingsException {
        m_saveOptimizerStateCheckbox.saveSettingsTo(settings);
        super.saveSettingsTo(settings);
        m_pythonCommandModel.saveSettingsTo(settings);
    }

    @Override
    protected void loadSettingsFrom(final NodeSettingsRO settings, final PortObjectSpec[] specs)
        throws NotConfigurableException {
        m_pythonCommandModel.loadSettingsFrom(settings);
        m_saveOptimizerStateCheckbox.loadSettingsFrom(settings, specs);
        super.loadSettingsFrom(settings, specs);
    }

    @Override
    public void onOpen() {
        m_pythonCommandModel.onDialogOpen();
    }
}
