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

import java.util.Arrays;

import org.knime.dl.core.DLFixedTensorShape;
import org.tensorflow.Tensor;

/**
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 * @param <T> The type of objects stored in this buffer.
 */
public abstract class TFAbstractTensorObjectBuffer <T> 
implements TFTensorWritableObjectBuffer<T>, TFTensorReadableObjectBuffer<T> {
	
	private final DLBytesConverter<T> m_converter;
	private TFUniversalWrappingObjectBuffer<byte[], ?> m_storage;
	
	/**
	 * @param bytesConverter
	 * @param shape
	 */
	public TFAbstractTensorObjectBuffer(DLBytesConverter<T> bytesConverter, long[] shape) {
		m_converter = bytesConverter;
		m_storage = DLBytesBuffers.createBytesBuffer(shape);
	}
	
	@Override
	public final long getCapacity() {
		return m_storage.getCapacity();
	}

	@Override
	public final void zeroPad(long length) {
		m_storage.zeroPad(length);
	}

	@Override
	public final void resetWrite() {
		m_storage.resetWrite();
	}

	@Override
	public final long size() {
		return m_storage.size();
	}

	@Override
	public final void close() {
		m_storage.close();
	}

	@Override
	public final void put(T value) {
		byte[] data = m_converter.toBytes(value);
		m_storage.put(data);
	}

	@Override
	public final void putAll(T[] values) {
		Arrays.stream(values).sequential()
		.map(m_converter::toBytes)
		.forEach(m_storage::put);
	}

	@Override
	public final void reset() {
		resetRead();
		resetWrite();
	}
	
	@Override
	public final Tensor<String> readIntoTensor(long batchSize, DLFixedTensorShape shape) {
		return Tensor.create(m_storage.getStorageForTensorCreation(batchSize), String.class);
	}

	@Override
	public final void writeFromTensor(Tensor<?> tensor) {
		tensor.copyTo(m_storage.getStorageForWriting(0, tensor.numElements()));
	}

	@Override
	public final void resetRead() {
		m_storage.resetRead();
	}

	@Override
	public final T readNext() {
		return m_converter.fromBytes(m_storage.readNext());
	}

}
