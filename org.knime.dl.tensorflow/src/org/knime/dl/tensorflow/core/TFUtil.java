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
package org.knime.dl.tensorflow.core;

import org.knime.core.node.NodeLogger;
import org.knime.core.util.Version;
import org.knime.dl.core.DLDimensionOrder;
import org.knime.dl.core.DLFixedTensorShape;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFUtil {

	private static final NodeLogger LOGGER = NodeLogger.getLogger(TFUtil.class);

	private TFUtil() {
		// Utility class
	}

	/** The default dimension order of TensorFlow */
	public static final DLDimensionOrder DEFAULT_DIMENSION_ORDER = DLDimensionOrder.TDHWC;

	/**
	 * Creates a shape array which can be used to create a {@link Tensor} from a batch size and a
	 * {@link DLFixedTensorShape}.
	 *
	 * @param batchSize the batch size of the tensor
	 * @param dlShape the shape of the tensor (without batch size)
	 * @return the shape of the tensor including the batch size in the first dimension
	 */
	public static long[] createTFShape(final long batchSize, final DLFixedTensorShape dlShape) {
		final long[] tfShape = new long[dlShape.getNumDimensions() + 1];
		tfShape[0] = batchSize;
		System.arraycopy(dlShape.getShape(), 0, tfShape, 1, dlShape.getNumDimensions());
		return tfShape;
	}

	/**
	 * Checks if the TensorFlow version of the network is compatible with the runtime TensorFlow version and logs a
	 * warning if it is not compatible.
	 *
	 * @param networkTFVersion the TensorFlow version of the network.
	 */
	public static void checkTFVersion(final Version networkTFVersion) {
		final Version runtimeTFVersion = new Version(TensorFlow.version());
		if (!runtimeTFVersion.isSameOrNewer(networkTFVersion)) {
			LOGGER.warn("The TensorFlow version of the network \"" + networkTFVersion.toString()
					+ "\" is newer than the runtime TensorFlow version \"" + runtimeTFVersion.toString() + "\".\n"
					+ "This could lead to unexpected behaviour.\n"
					+ "If the network has been created by the Python Network Creator or the TensorFlow Converter "
					+ "this could mean that your Python TensorFlow version is to new.");
		}
	}
}
