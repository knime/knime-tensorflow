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
package org.knime.dl.tensorflow.core.convert;

import java.util.HashMap;
import java.util.Map;

import org.eclipse.core.runtime.IConfigurationElement;
import org.knime.dl.core.DLAbstractExtensionPointRegistry;
import org.knime.dl.core.DLNetwork;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFNetworkConverterRegistry extends DLAbstractExtensionPointRegistry {

	private static final String EXT_POINT_ID = "org.knime.dl.tensorflow.TFNetworkConverter";

	private static final String EXT_POINT_ATTR_CLASS = "TFNetworkConverter";

	private static TFNetworkConverterRegistry instance;

	/**
	 * @return singleton instance of the {@link TFNetworkConverterRegistry}
	 */
	public static synchronized TFNetworkConverterRegistry getInstance() {
		if (instance == null) {
			instance = new TFNetworkConverterRegistry();
            // First set instance, then register. Registering usually activates other bundles. Those may try to access
            // this registry (while the instance is still null) which would trigger another instance construction.
            instance.register();
		}
		return instance;
	}

	private final Map<Class<? extends DLNetwork>, TFNetworkConverter> m_converters = new HashMap<>();

	private TFNetworkConverterRegistry() {
		super(EXT_POINT_ID, EXT_POINT_ATTR_CLASS);
		// Do not trigger registration here. See #getInstance() above.
	}

	/**
	 * Finds the appropriate converter for the given network type.
	 *
	 * @param networkType the deep-learning network type
	 * @return the converter
	 */
	public TFNetworkConverter getConverter(final Class<? extends DLNetwork> networkType) {
		// TODO should we allow super classes?
		return m_converters.get(networkType);
	}

	@Override
	protected void registerInternal(final IConfigurationElement elem, final Map<String, String> attrs)
			throws Throwable {
		registerConverterInternal((TFNetworkConverter) elem.createExecutableExtension(EXT_POINT_ATTR_CLASS));
	}

	private synchronized void registerConverterInternal(final TFNetworkConverter converter) {
		final Class<? extends DLNetwork> networkType = converter.getNetworkType();
		if (networkType == null) {
			throw new IllegalArgumentException("The converter's associated network type must not be null.");
		}
		m_converters.put(networkType, converter);
	}
}
