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
package org.knime.dl.tensorflow.savedmodel.core.execution;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

import org.knime.core.util.FileUtil;
import org.knime.dl.core.DLCanceledExecutionException;
import org.knime.dl.core.DLNetworkInputPreparer;
import org.knime.dl.core.DLTensor;
import org.knime.dl.core.DLTensorFactory;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.data.DLReadableBuffer;
import org.knime.dl.core.data.DLWrappingDataBuffer;
import org.knime.dl.core.data.DLWritableBuffer;
import org.knime.dl.core.execution.DLAbstractNetworkExecutionSession;
import org.knime.dl.core.execution.DLExecutionMonitor;
import org.knime.dl.core.execution.DLNetworkOutputConsumer;
import org.knime.dl.tensorflow.core.execution.DLTensorFlowNetworkExecutionSession;
import org.knime.dl.tensorflow.savedmodel.core.DLTensorFlowSavedModelNetwork;
import org.knime.dl.util.DLUtils;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowSavedModelNetworkExecutionSession extends
	DLAbstractNetworkExecutionSession<DLTensorFlowSavedModelNetwork> implements DLTensorFlowNetworkExecutionSession {

	private SavedModelBundle m_savedModelBundle;

	protected DLTensorFlowSavedModelNetworkExecutionSession(final DLTensorFlowSavedModelNetwork network,
			final Set<DLTensorSpec> executionInputSpecs, final Set<DLTensorId> requestedOutputs,
			final DLNetworkInputPreparer inputPreparer, final DLNetworkOutputConsumer outputConsumer,
			final DLTensorFactory tensorFactory) {
		super(network, executionInputSpecs, requestedOutputs, inputPreparer, outputConsumer, tensorFactory);
	}

	@Override
	protected void executeInternal(final DLExecutionMonitor monitor) throws DLCanceledExecutionException, Exception {
		if (m_savedModelBundle == null) {
			m_savedModelBundle = SavedModelBundle.load(FileUtil.getFileFromURL(m_network.getSource()).getAbsolutePath(),
					m_network.getSpec().getTags());
		}
		try (final Session session = m_savedModelBundle.session()) {
			final Runner runner = session.runner();

			// Feed the inputs
			for (final Entry<DLTensorId, DLTensor<? extends DLWritableBuffer>> e : m_input.entrySet()) {
				runner.feed(e.getKey().getIdentifierString(), createTFTensor(e.getValue()));
			}

			// Fetch the outputs
			final List<Entry<DLTensorId, DLTensor<? extends DLReadableBuffer>>> outputs = new ArrayList<>(
					m_output.entrySet());
			for (int i = 0; i < outputs.size(); i++) {
				runner.fetch(outputs.get(i).getKey().getIdentifierString(), i);
			}

			// Run the model
			final List<Tensor<?>> runOutputs = runner.run();

			// Write the result to the KNIME tensors
			for (int i = 0; i < runOutputs.size(); i++) {
				writeToDLTensor(runOutputs.get(i), outputs.get(i).getValue());
			}
		}
	}

	@SuppressWarnings("unchecked")
	private Tensor<?> createTFTensor(final DLTensor<? extends DLWritableBuffer> dlTensor) {
		// TODO throw exception
		final long[] shape = Arrays.stream(DLUtils.Shapes.getFixedShape(dlTensor.getSpec().getShape()).get()).skip(1)
				.toArray();
		final Class<?> elementType = dlTensor.getSpec().getElementType();

		Tensor<?> tfTensor;
		if (elementType.equals(int.class)) {
			final DLWrappingDataBuffer<int[]> dlBuffer = (DLWrappingDataBuffer<int[]>) dlTensor.getBuffer();
			tfTensor = Tensor.create(shape, IntBuffer.wrap(dlBuffer.getStorageForReading(0, dlBuffer.size())));
		} else if (elementType.equals(long.class)) {
			final DLWrappingDataBuffer<long[]> dlBuffer = (DLWrappingDataBuffer<long[]>) dlTensor.getBuffer();
			tfTensor = Tensor.create(shape, LongBuffer.wrap(dlBuffer.getStorageForReading(0, dlBuffer.size())));
		} else if (elementType.equals(float.class)) {
			final DLWrappingDataBuffer<float[]> dlBuffer = (DLWrappingDataBuffer<float[]>) dlTensor.getBuffer();
			tfTensor = Tensor.create(shape, FloatBuffer.wrap(dlBuffer.getStorageForReading(0, dlBuffer.size())));
		} else if (elementType.equals(double.class)) {
			final DLWrappingDataBuffer<double[]> dlBuffer = (DLWrappingDataBuffer<double[]>) dlTensor.getBuffer();
			tfTensor = Tensor.create(shape, DoubleBuffer.wrap(dlBuffer.getStorageForReading(0, dlBuffer.size())));
		} else {
			throw new IllegalStateException("The data type " + elementType + "is not supported.");
		}
		return tfTensor;
	}

	@SuppressWarnings("unchecked")
	private void writeToDLTensor(final Tensor<?> tfTensor, final DLTensor<? extends DLReadableBuffer> dlTensor) {
		// TODO how to tell the KNIME tensor the shape?
		switch (tfTensor.dataType()) {
		case INT32:
			final IntBuffer intBuffer = IntBuffer.allocate(tfTensor.numElements());
			tfTensor.writeTo(intBuffer);
			((DLWrappingDataBuffer<int[]>) dlTensor.getBuffer()).setStorage(intBuffer.array(), intBuffer.capacity());
			return;
		case INT64:
			final LongBuffer longBuffer = LongBuffer.allocate(tfTensor.numElements());
			tfTensor.writeTo(longBuffer);
			((DLWrappingDataBuffer<long[]>) dlTensor.getBuffer()).setStorage(longBuffer.array(), longBuffer.capacity());
			return;
		case FLOAT:
			final FloatBuffer floatBuffer = FloatBuffer.allocate(tfTensor.numElements());
			tfTensor.writeTo(floatBuffer);
			((DLWrappingDataBuffer<float[]>) dlTensor.getBuffer()).setStorage(floatBuffer.array(),
					floatBuffer.capacity());
			return;
		case DOUBLE:
			final DoubleBuffer doubleBuffer = DoubleBuffer.allocate(tfTensor.numElements());
			tfTensor.writeTo(doubleBuffer);
			((DLWrappingDataBuffer<double[]>) dlTensor.getBuffer()).setStorage(doubleBuffer.array(),
					doubleBuffer.capacity());
			return;
		default:
			throw new IllegalStateException("Tensor has unsupported type " + tfTensor.dataType());
		}
	}
}
