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
package org.knime.tensorflow.libs;

import java.io.File;
import java.io.IOException;
import java.net.URL;

import org.eclipse.core.runtime.FileLocator;
import org.eclipse.core.runtime.Path;
import org.eclipse.core.runtime.Platform;
import org.eclipse.ui.plugin.AbstractUIPlugin;
import org.knime.core.node.NodeLogger;
import org.knime.core.util.FileUtil;
import org.knime.tensorflow.libs.prefs.TFPreferencePage;
import org.osgi.framework.Bundle;
import org.osgi.framework.BundleContext;

/**
 * Activator class for the TensorFlow libraries.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFPluginActivator extends AbstractUIPlugin {

	private static final String GPU_FRAGMENT_REGEX = "org\\.knime\\.tensorflow\\.bin\\.[^\\.]+\\.amd64\\.gpu.*";

	private static final String CPU_FRAGMENT_REGEX = "org\\.knime\\.tensorflow\\.bin\\.[^\\.]+\\.amd64\\.cpu.*";

	private static final NodeLogger LOGGER = NodeLogger.getLogger(TFPluginActivator.class);

	/** The plug-in ID */
	public static final String PLUGIN_ID = "org.knime.tensorflow.libs"; //$NON-NLS-1$

	// The shared instance
	private static TFPluginActivator plugin;

	@Override
	public void start(final BundleContext context) throws Exception {
		super.start(context);
		plugin = this;
		
		String osName = System.getProperty("os.name");
		String osVersion = System.getProperty("os.version");
		if(osName.toLowerCase().contains("mac")) {
			if(osVersion.toLowerCase().contains("10.11")) {
				Exception ex = new IllegalStateException("Can't start KNIME Deep Learning - Tensorflow Integration. "
						+ "macOs 10.11 (El Capitan) is not supported by Tensorflow. Please use "
						+ "macOS 10.12.6 (Sierra) or later.");
				LOGGER.error(ex.getMessage());
				throw ex;
			}
		}

		initPreferencePageDefaults();
		final boolean forceCPU = plugin.getPreferenceStore().getBoolean(TFPreferencePage.P_FORCE_CPU);

		// Get the path to the native libraries for GPU and CPU
		String gpuLibPath = null;
		String cpuLibPath = null;
		final Bundle[] fragments = Platform.getFragments(getBundle());
		for (Bundle fragment : fragments) {
			final String symbolicName = fragment.getSymbolicName();
			if (!forceCPU && symbolicName.matches(GPU_FRAGMENT_REGEX)) {
				gpuLibPath = getNativeLibPath(fragment);
			} else if (symbolicName.matches(CPU_FRAGMENT_REGEX)) {
				cpuLibPath = getNativeLibPath(fragment);
			}
		}

		// Load one of the native libraries
		loadLibrary(gpuLibPath, cpuLibPath);
	}

	@Override
	public void stop(final BundleContext context) throws Exception {
		plugin = null;
		super.stop(context);
	}

	/**
	 * Returns the shared instance
	 *
	 * @return the shared instance
	 */
	public static TFPluginActivator getDefault() {
		return plugin;
	}

	private String getNativeLibPath(final Bundle fragmentBundle) throws IOException {
		final URL libsUrl = FileLocator.find(fragmentBundle, new Path("libs/"), null);
		final File libsFile = FileUtil.getFileFromURL(FileLocator.toFileURL(libsUrl));
		final File nativeLib = libsFile.listFiles((f, s) -> s.contains("tensorflow_jni"))[0];
		return nativeLib.getAbsolutePath();
	}

	private void loadLibrary(final String gpuLib, final String cpuLib) {
		if (gpuLib != null) {
			// Try to load the GPU version of TensorFlow
			// This will only succeed if there is a valid CUDA installation
			try {
				System.load(gpuLib);
				return;
			} catch (UnsatisfiedLinkError e) {
				LOGGER.info("Could not load TensorFlow GPU library.", e);
			}
		}
		if (cpuLib != null) {
			// This should work always
			System.load(cpuLib);
			return;
		}
		// We couldn't load any of the libraries
		throw new IllegalStateException(
				"Could not load the TensorFlow JNI from a fragment. This is most likely an implementation error.");
	}

	private void initPreferencePageDefaults() {
		plugin.getPreferenceStore().setDefault(TFPreferencePage.P_FORCE_CPU,
				TFPreferencePage.DEFAULT_FORCE_CPU);
	}
}
