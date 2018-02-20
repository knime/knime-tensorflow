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
package org.knime.dl.tensorflow.savedmodel.core.data;

import java.nio.DoubleBuffer;

import org.knime.dl.core.DLFixedTensorShape;
import org.knime.dl.core.DLInvalidNetworkInputException;
import org.knime.dl.core.DLInvalidNetworkOutputException;
import org.knime.dl.core.data.DLDefaultDoubleBuffer;
import org.knime.dl.tensorflow.core.DLTensorFlowUtil;
import org.tensorflow.DataType;
import org.tensorflow.Tensor;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowTensorDoubleBuffer extends DLDefaultDoubleBuffer
		implements DLTensorFlowTensorReadableDoubleBuffer, DLTensorFlowTensorWritableDoubleBuffer {

	/**
	 * Creates a new instance of this buffer.
	 *
	 * @param capacity the immutable capacity of the buffer
	 */
	public DLTensorFlowTensorDoubleBuffer(final long capacity) {
		super(capacity);
	}

	@Override
	public Tensor<Double> readIntoTensor(final long batchSize, final DLFixedTensorShape shape)
			throws DLInvalidNetworkInputException {
		final long[] tfShape = DLTensorFlowUtil.createTFShape(batchSize, shape);
		final int bufferSize;
		try {
			bufferSize = Math.toIntExact(size());
		} catch (final ArithmeticException e) {
			throw new DLInvalidNetworkInputException("Tried to create a TensorFlow tensor with " + size()
					+ " elements but a TensorFlow tensor can only contain " + Integer.MAX_VALUE + " elements.");
		}
		return Tensor.create(tfShape, DoubleBuffer.wrap(getStorageForReading(0, bufferSize), 0, bufferSize));
	}

	@Override
	public void writeFromTensor(final Tensor<?> tensor) throws DLInvalidNetworkOutputException {
		if (tensor.dataType() != DataType.DOUBLE) {
			throw new DLInvalidNetworkOutputException("Writing a TensorFlow tensor of type " + tensor.dataType()
					+ " to a double buffer is not supported.");
		}
		final DoubleBuffer doubleBuffer = DoubleBuffer.wrap(getStorageForWriting(0, tensor.numElements()));
		tensor.writeTo(doubleBuffer);
	}
}
