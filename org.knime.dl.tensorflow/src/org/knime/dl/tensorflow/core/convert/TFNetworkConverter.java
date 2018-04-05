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

import org.knime.core.data.filestore.FileStore;
import org.knime.dl.core.DLNetwork;
import org.knime.dl.core.DLNetworkSpec;
import org.knime.dl.tensorflow.core.TFNetwork;
import org.knime.dl.tensorflow.core.TFNetworkSpec;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public interface TFNetworkConverter {

	/**
	 * @return Type of the deep-learning network this converter can handle
	 */
	Class<? extends DLNetwork> getNetworkType();

	/**
	 * @return Type of the TensorFlow network this converter creates.
	 */
	Class<? extends TFNetwork> getOutputNetworkType();

	/**
	 * @param specType type of the network specs
	 * @return true if the converter can convert this type of network spec
	 */
	boolean canConvertSpec(Class<? extends DLNetworkSpec> specType);

	/**
	 * Converts the network spec of a deep-learning network if converting specs is supported. Call
	 * {@link #canConvertSpec(Class)} to find out if converting network specs is supported.
	 *
	 * @param spec the deep-learning network spec
	 * @return the appropriate TensorFlow deep-learning network spec
	 * @throws DLNetworkConversionException if converting the specs failed
	 */
	TFNetworkSpec convertSpec(DLNetworkSpec spec) throws DLNetworkConversionException;

	/**
	 * Converts the deep-learning network to a TensorFlow deep-learning network.
	 *
	 * @param network the deep-learning network
	 * @param fileStore a file store to store the TensorFlow deep-learning network in
	 * @return the converted TensorFlow deep-learning network.
	 * @throws DLNetworkConversionException if converting the network failed.
	 */
	TFNetwork convertNetwork(DLNetwork network, FileStore fileStore) throws DLNetworkConversionException;
}
