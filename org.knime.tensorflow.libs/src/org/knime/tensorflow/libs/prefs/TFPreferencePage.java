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
package org.knime.tensorflow.libs.prefs;

import org.eclipse.jface.preference.BooleanFieldEditor;
import org.eclipse.jface.preference.FieldEditorPreferencePage;
import org.eclipse.jface.preference.IPreferenceStore;
import org.eclipse.swt.SWT;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.MessageBox;
import org.eclipse.ui.IWorkbench;
import org.eclipse.ui.IWorkbenchPreferencePage;
import org.eclipse.ui.PlatformUI;
import org.knime.tensorflow.libs.TFPluginActivator;

/**
 * Preference page for the TensorFlow deep learning backend.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFPreferencePage extends FieldEditorPreferencePage implements IWorkbenchPreferencePage {

	/** Key for the CPU checkbox. */
	public static final String P_FORCE_CPU = "forceCPU";

	/** Default value if CPU usage should be forced. */
	public static boolean DEFAULT_FORCE_CPU = false;

	private boolean m_forceCPU;

	/**
	 * Constructor for a TensorFlow preference page.
	 */
	public TFPreferencePage() {
		super(GRID);
		IPreferenceStore ps = TFPluginActivator.getDefault().getPreferenceStore();
		setPreferenceStore(ps);
		setDescription("Preferences for the KNIME - TensorFlow Integration.");
		m_forceCPU = ps.getBoolean(P_FORCE_CPU);
	}

	@Override
	protected void createFieldEditors() {
		addField(new LabelField(getFieldEditorParent(),
				"The TensorFlow integration tries to load the TensorFlow GPU binaries by default.\n"
						+ "If this fails the TensorFlow CPU binaries are loaded.\n"
						+ "Note that TensorFlow GPU is currently only available for Linux.\n"
						+ "Therefore this setting will have no effect on Windows or MacOS."));
		addField(new BooleanFieldEditor(P_FORCE_CPU, "Force CPU", BooleanFieldEditor.SEPARATE_LABEL,
				getFieldEditorParent()));
	}

	@Override
	protected void performApply() {
		// Call super method to not show restart dialog
		super.performOk();
	}

	@Override
	public boolean performOk() {
		boolean result = super.performOk();
		checkChanges();
		return result;
	}

	private void checkChanges() {
		boolean currentForceCPU = getPreferenceStore().getBoolean(P_FORCE_CPU);
		if (m_forceCPU != currentForceCPU) {
			m_forceCPU = currentForceCPU;
			Display.getDefault().asyncExec(
					() -> promptRestartWithMessage("Changes become first available after restarting the workbench.\n"
							+ "Do you want to restart the workbench now?"));
		}
	}

	private static void promptRestartWithMessage(final String message) {
		MessageBox mb = new MessageBox(Display.getDefault().getActiveShell(), SWT.ICON_QUESTION | SWT.YES | SWT.NO);
		mb.setText("Restart workbench...");
		mb.setMessage(message);
		if (mb.open() != SWT.YES) {
			return;
		}
		PlatformUI.getWorkbench().restart();
	}

	@Override
	public void init(IWorkbench workbench) {
		// nothing to do
	}
}
