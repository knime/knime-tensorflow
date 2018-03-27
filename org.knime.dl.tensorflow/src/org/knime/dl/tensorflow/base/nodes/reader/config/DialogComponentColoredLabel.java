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

import java.awt.Color;
import java.awt.FlowLayout;

import javax.swing.JLabel;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.defaultnodesettings.DialogComponent;
import org.knime.core.node.port.PortObjectSpec;

/**
 * Dialog component containing only a label which can be colored.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DialogComponentColoredLabel extends DialogComponent {

	private final JLabel m_label;

	/**
	 * Creates a dialog component containing only a colored label.
	 *
	 * @param text the text of the label
	 * @param color the text color of the label
	 * @param width the width of the label
	 */
	public DialogComponentColoredLabel(final String text, final Color color, final int width) {
		super(new EmptySettingsModel());

		m_label = new JLabel("<html><div WIDTH=" + width + ">" + text + "</div></html>");
		m_label.setForeground(color);
		getComponentPanel().setLayout(new FlowLayout());
		getComponentPanel().add(m_label);
	}

	/**
	 * Sets the text of the label.
	 *
	 * @param text the text
	 */
	public void setText(final String text) {
		m_label.setText("<html><div WIDTH=400>" + text + "</div></html>");
	}

	/**
	 * Sets the text color of the label.
	 *
	 * @param color the text color
	 */
	public void setColor(final Color color) {
		m_label.setForeground(color);
	}

	@Override
	protected void updateComponent() {
		// Nothing to do
	}

	@Override
	protected void validateSettingsBeforeSave() throws InvalidSettingsException {
		// Nothing to do
	}

	@Override
	protected void checkConfigurabilityBeforeLoad(final PortObjectSpec[] specs) throws NotConfigurableException {
		// Nothing to do
	}

	@Override
	protected void setEnabledComponents(final boolean enabled) {
		m_label.setEnabled(enabled);
	}

	@Override
	public void setToolTipText(final String text) {
		m_label.setToolTipText(text);
	}

}
