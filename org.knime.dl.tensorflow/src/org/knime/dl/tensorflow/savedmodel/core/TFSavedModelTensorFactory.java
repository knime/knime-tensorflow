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
package org.knime.dl.tensorflow.savedmodel.core;

import java.util.function.Supplier;

import org.knime.dl.core.DLDefaultFixedTensorShape;
import org.knime.dl.core.DLDefaultTensor;
import org.knime.dl.core.DLDefaultTensorSpec;
import org.knime.dl.core.DLFixedTensorShape;
import org.knime.dl.core.DLTensor;
import org.knime.dl.core.DLTensorFactory;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.data.DLBuffer;
import org.knime.dl.core.data.DLReadableBuffer;
import org.knime.dl.core.data.DLWritableBuffer;
import org.knime.dl.tensorflow.core.TFUtil;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorBitBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorDoubleBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorFloatBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorIntBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorLongBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorReadableBitBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorReadableDoubleBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorReadableFloatBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorReadableIntBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorReadableLongBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorReadableStringBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorReadableUnsignedByteBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorStringBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorUnsignedByteBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorWritableBitBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorWritableDoubleBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorWritableFloatBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorWritableIntBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorWritableLongBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorWritableStringBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorWritableUnsignedByteBuffer;
import org.knime.dl.util.DLUtils;

import com.google.common.primitives.UnsignedBytes;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public class TFSavedModelTensorFactory implements DLTensorFactory {

	@Override
	public Class<? extends DLWritableBuffer> getWritableBufferType(final DLTensorSpec spec) {
		final Class<?> t = spec.getElementType();
		if (t.equals(double.class)) {
			return TFTensorWritableDoubleBuffer.class;
		} else if (t.equals(float.class)) {
			return TFTensorWritableFloatBuffer.class;
		} else if (t.equals(boolean.class)) {
			return TFTensorWritableBitBuffer.class;
		} else if (t.equals(UnsignedBytes.class)) {
			return TFTensorWritableUnsignedByteBuffer.class;
		} else if (t.equals(int.class)) {
			return TFTensorWritableIntBuffer.class;
		} else if (t.equals(long.class)) {
			return TFTensorWritableLongBuffer.class;
		} else if (t.equals(String.class)) {
			return TFTensorWritableStringBuffer.class;
		} else {
			throw new IllegalArgumentException("No matching buffer type.");
		}
	}

	@Override
	public Class<? extends DLReadableBuffer> getReadableBufferType(final DLTensorSpec spec) {
		final Class<?> t = spec.getElementType();
		if (t.equals(double.class)) {
			return TFTensorReadableDoubleBuffer.class;
		} else if (t.equals(float.class)) {
			return TFTensorReadableFloatBuffer.class;
		} else if (t.equals(boolean.class)) {
			return TFTensorReadableBitBuffer.class;
		} else if (t.equals(UnsignedBytes.class)) {
			return TFTensorReadableUnsignedByteBuffer.class;
		} else if (t.equals(int.class)) {
			return TFTensorReadableIntBuffer.class;
		} else if (t.equals(long.class)) {
			return TFTensorReadableLongBuffer.class;
		} else if (t.equals(String.class)) {
			return TFTensorReadableStringBuffer.class;
		} else {
			throw new IllegalArgumentException("No matching buffer type.");
		}
	}

	@Override
	public DLTensor<? extends DLWritableBuffer> createWritableTensor(final DLTensorSpec spec) {
		return createTensorInternal(spec);
	}

	@Override
	public DLTensor<? extends DLReadableBuffer> createReadableTensor(final DLTensorSpec spec) {
		return createTensorInternal(spec);
	}

	@Override
	public DLTensorSpec createExecutionTensorSpec(final DLTensorSpec spec, final long batchSize, final long[] shape) {
		return new DLDefaultTensorSpec(spec.getIdentifier(), spec.getName(), batchSize,
				new DLDefaultFixedTensorShape(shape), spec.getElementType(), spec.getDimensionOrder());
	}

	@SuppressWarnings("unchecked")
	private <B extends DLBuffer> DLTensor<B> createTensorInternal(final DLTensorSpec spec) {
		final long[] shape = DLUtils.Shapes.getFixedShape(spec.getShape())
				.orElseThrow(() -> new IllegalArgumentException(
						"Tensor spec '" + spec.getName() + "' does not provide a shape. Tensor cannot be created."));
		if (!spec.getBatchSize().isPresent()) {
			throw new IllegalArgumentException(
					"Tensor spec '" + spec.getName() + "' does not provide a batch size. Tensor cannot be created.");
		}
		final long exampleSize = DLUtils.Shapes.getSize(shape);
		final long batchSize = spec.getBatchSize().getAsLong();
		final long size = exampleSize * batchSize;
		final Class<?> t = spec.getElementType();
		// TODO: handle unsafe casts
		final Supplier<B> s;
		if (t.equals(double.class)) {
			s = () -> (B) new TFTensorDoubleBuffer(size);
		} else if (t.equals(float.class)) {
			s = () -> (B) new TFTensorFloatBuffer(size);
		} else if (t.equals(boolean.class)) {
			s = () -> (B) new TFTensorBitBuffer(size);
		} else if (t.equals(UnsignedBytes.class)) {
			s = () -> (B) new TFTensorUnsignedByteBuffer(size);
		} else if (t.equals(int.class)) {
			s = () -> (B) new TFTensorIntBuffer(size);
		} else if (t.equals(long.class)) {
			s = () -> (B) new TFTensorLongBuffer(size);
		} else if (t.equals(String.class)) {
			s = () -> (B) new TFTensorStringBuffer(
					TFUtil.createTFShape(batchSize, (DLFixedTensorShape) spec.getShape()));
		} else {
			throw new IllegalArgumentException("No matching tensor type for tensor spec '" + spec.getName() + "'.");
		}
		return new DLDefaultTensor<>(spec, s.get(), exampleSize);
	}
}
