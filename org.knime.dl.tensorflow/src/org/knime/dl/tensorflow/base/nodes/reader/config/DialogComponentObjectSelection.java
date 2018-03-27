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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import java.awt.Component;
import java.awt.Dimension;
import java.util.Collection;
import java.util.function.BiConsumer;
import java.util.function.Function;

import javax.swing.DefaultListCellRenderer;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JList;

import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NotConfigurableException;
import org.knime.core.node.defaultnodesettings.DialogComponent;
import org.knime.core.node.defaultnodesettings.SettingsModel;
import org.knime.core.node.port.PortObjectSpec;

/**
 * DialogComponent for selecting arbitrary objects.
 * <p>
 * TODO Uses much code and should be merged with {@link org.knime.dl.base.nodes.DialogComponentObjectSelection}.
 *
 * @see org.knime.dl.base.nodes.DialogComponentObjectSelection
 * @param <T> the type of the SettingsModel
 * @param <R> the item type of the selection component
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DialogComponentObjectSelection<T extends SettingsModel, R> extends DialogComponent {

	private final Function<? super R, String> m_printer;

	private final BiConsumer<R, T> m_modelUpdater;

	private final Function<T, R> m_selectionUpdater;

	private final JComboBox<R> m_combobox;

	private final JLabel m_label;

	private boolean m_isReplacing;

	/**
	 * Creates a new instance of this dialog component.
	 *
	 * @param model the SettingsModel which stores the selected object
	 * @param printer a function that turns the component's items into a renderable string representation
	 * @param modelUpdater a function which sets the selected object in the SettingsModel
	 * @param selectionUpdater a function which reads the saved object from the SettingsModel
	 * @param label the label of the component
	 */
	public DialogComponentObjectSelection(final T model, final Function<? super R, String> printer,
			final BiConsumer<R, T> modelUpdater, final Function<T, R> selectionUpdater, final String label) {
		super(model);
		m_printer = printer;
		m_modelUpdater = modelUpdater;
		m_selectionUpdater = selectionUpdater;

		// Build panel
		m_label = new JLabel(label);
		getComponentPanel().add(m_label);
		m_combobox = new JComboBox<>();
		getComponentPanel().add(m_combobox);

		// String renderer for R
		final DefaultListCellRenderer renderer = new DefaultListCellRenderer() {
			private static final long serialVersionUID = 1L;

			@SuppressWarnings("unchecked")
			@Override
			public Component getListCellRendererComponent(final JList<?> list, final Object value, final int index,
					final boolean isSelected, final boolean cellHasFocus) {
				return super.getListCellRendererComponent(list, value != null ? m_printer.apply((R) value) : "", index,
						isSelected, cellHasFocus);
			}
		};
		m_combobox.setRenderer(renderer);

		// Selection changes
		m_combobox.addActionListener(e -> updateModel());

		// Config changes
		getModel().addChangeListener(e -> updateComponent());
	}

	/**
	 * Replaces the list of selectable objects in the component. If <code>selected</code> is specified (not null) and it
	 * exists in the collection, it will be selected. If <code>selected</code> is null, the entry that corresponds to
	 * the previous hidden's value will stay selected (if it exists in the new list).
	 *
	 * @param newItems the items that will be displayed in the dialog component. No null values, no duplicate values.
	 *            Must be at least of length one.
	 * @param selected the item to select after replacing. Can be null, in which case the previous selection is tried to
	 *            be preserved.
	 */
	@SuppressWarnings("unchecked")
	public void replaceListItems(final Collection<R> newItems, final R selected) {
		checkNotNull(newItems);
		checkArgument(!newItems.isEmpty());
		final R newItem = selected != null ? selected : m_selectionUpdater.apply((T) getModel());

		m_isReplacing = true;
		m_combobox.removeAllItems();
		for (final R item : newItems) {
			m_combobox.addItem(item);
		}
		if (newItem != null && newItems.contains(newItem)) {
			m_combobox.setSelectedItem(newItem);
		} else {
			m_combobox.setSelectedIndex(0);
		}
		m_isReplacing = false;

		m_combobox.setSize(m_combobox.getPreferredSize());
		getComponentPanel().validate();
		updateModel();
	}

	@SuppressWarnings("unchecked")
	@Override
	protected void updateComponent() {
		final R newValue = m_selectionUpdater.apply((T) getModel());
		final R oldValue = (R) m_combobox.getSelectedItem();
		final boolean updateSelection;
		if (newValue == null) {
			updateSelection = oldValue != null;
		} else {
			updateSelection = !newValue.equals(oldValue);
		}
		if (updateSelection) {
			m_combobox.setSelectedItem(newValue);
		}
		setEnabledComponents(getModel().isEnabled());
		final R newValueAfterUpdate = (R) m_combobox.getSelectedItem();
		final boolean selectionChanged;
		if (newValueAfterUpdate == null) {
			selectionChanged = newValue != null;
		} else {
			selectionChanged = !newValueAfterUpdate.equals(newValue);
		}
		if (selectionChanged) {
			updateModel();
		}
	}

	@SuppressWarnings("unchecked")
	private void updateModel() {
		if (m_isReplacing) {
			return;
		}
		final R newValue = (R) m_combobox.getSelectedItem();
		m_modelUpdater.accept(newValue, (T) getModel());
	}

	@Override
	protected void validateSettingsBeforeSave() throws InvalidSettingsException {
		updateModel();
	}

	@Override
	protected void checkConfigurabilityBeforeLoad(final PortObjectSpec[] specs) throws NotConfigurableException {
		// nothing to do

	}

	@Override
	protected void setEnabledComponents(final boolean enabled) {
		m_combobox.setEnabled(enabled);
	}

	/**
	 * Sets the preferred size of the internal {@link JComboBox}.
	 *
	 * @param width The width
	 * @param height The height
	 */
	public void setSizeComboBox(final int width, final int height) {
		m_combobox.setPreferredSize(new Dimension(width, height));
	}

	/**
	 * Sets the prefered size of the internal label.
	 *
	 * @param width The width
	 * @param height The height
	 */
	public void setSizeLabel(final int width, final int height) {
		m_label.setPreferredSize(new Dimension(width, height));
	}

	@Override
	public void setToolTipText(final String text) {
		m_label.setToolTipText(text);
		m_combobox.setToolTipText(text);
	}
}
