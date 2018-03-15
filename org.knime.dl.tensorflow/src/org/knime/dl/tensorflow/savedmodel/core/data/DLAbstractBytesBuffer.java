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

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Arrays;

import org.knime.dl.core.data.DLAbstractWrappingDataBuffer;
import org.knime.dl.util.DLUtils;

/**
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 * @param <S> the type of multi-dimensional storage array
 */
abstract class DLAbstractBytesBuffer <S> extends DLAbstractWrappingDataBuffer<S> 
implements TFUniversalWrappingObjectBuffer<byte[], S> {

	private static final byte[] ZERO_VALUE = new byte[0];
	
	private final int[] m_shape;
	private final int[] m_tmpPosition;
	
	/**
	 * @param shape
	 */
	protected DLAbstractBytesBuffer(long[] shape) {
		super(DLUtils.Shapes.getSize(shape));
		try {
			m_shape = Arrays.stream(shape).mapToInt(Math::toIntExact).toArray();
		} catch (ArithmeticException e) {
			// can't happen because super constructor already checks the capacity of
			// the complete buffer (which is <= a single dimension) but check anyway
			throw new IllegalArgumentException(
					"Currently a dimension in a shape may not exceed Integer.MAX_VALUE.");
		}
		m_tmpPosition = new int[m_shape.length];
	}
	
	@Override
	public final void zeroPad(long length) {
		checkArgument(length > 0);
		checkOverflow(m_nextWrite + length <= m_capacity);
		for (int i = 0; i < length; i++) {
			putInternal(ZERO_VALUE);
		}
	}
	
	@Override
	public final S getStorageForTensorCreation(long batchSize) {
		S storage = getStorageForReading(0, batchSize);
		if (batchSize == m_shape[0]) {
			return storage;
		}
		return createSubArray(storage, (int) batchSize);
	}
	
	private S createSubArray(S storage, int batchSize) {
		S subArray = createEmptySubArray(batchSize);
		System.arraycopy(storage, 0, subArray, 0, batchSize);
		return subArray;
	}
	
	@Override
	public final void setStorage(S storage, long storageSize) {
		checkArgument(getLength(storage) == m_capacity, "Input storage capacity does not match buffer capacity.");
		setStorage(storage);
		m_nextWrite = (int) storageSize;
		resetRead();
	}
	
	@Override
	protected final S createStorage() {
		return createStorage(m_shape);
	}
	
	@Override
	public final void put(byte[] value) {
		checkOverflow(m_nextWrite < m_capacity);
		putInternal(value);
	}
	
	private void putInternal(byte[] value) {
		int[] position = calculatePositionInStorage(m_nextWrite++);
		placeInStorage(value, position);
	}
	
	@Override
	public final void putAll(byte[][] values) {
		checkOverflow(m_nextWrite + values.length <= m_capacity);
		for (byte[] value : values) {
			putInternal(value);
		}
	}
	
	@Override
	public final byte[] readNext() {
		checkUnderflow(m_nextRead < m_nextWrite);
		int[] position = calculatePositionInStorage(m_nextRead++);
 		return retrieveFromStorage(position);
	}
	
	/**
	 * @param position in multi-dimensional array
	 * @return value at <b>position</b>
	 */
	protected abstract byte[] retrieveFromStorage(int[] position);
	
	/**
	 * @param value the value to place at <b>position</b>
	 * @param position in the multi-dimensional array 
	 * 
	 */
	protected abstract void placeInStorage(byte[] value, int[] position);
	
	/**
	 * @param storage to get length of
	 * @return length of <b>storage</b>
	 * 
	 */
	protected abstract long getLength(S storage);
	
	protected abstract S createEmptySubArray(int length);  
	
	/**
	 * @param shape the shape of the multi-dimensional storage array
	 * @return multi-dimensional storage array
	 * 
	 */
	protected abstract S createStorage(int[] shape);
	
	private final int[] calculatePositionInStorage(int flatIdx) {
		int currentFlatIdx = flatIdx;
		for (int i = m_shape.length - 1; i >= 0; i--) {
			m_tmpPosition[i] = currentFlatIdx % m_shape[i];
			currentFlatIdx = roundUp(currentFlatIdx, m_shape[i]);
		}
		return m_tmpPosition;
	}
	
	private static int roundUp(int num, int divisor) {
	    return (num + divisor - 1) / divisor;
	}

	@Override
	public void reset() {
		resetRead();
		resetWrite();
	}
	
}
