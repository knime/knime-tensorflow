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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.ArrayUtils;
import org.knime.dl.core.DLCanceledExecutionException;
import org.knime.dl.core.DLFixedTensorShape;
import org.knime.dl.core.DLNetworkInputPreparer;
import org.knime.dl.core.DLTensor;
import org.knime.dl.core.DLTensorFactory;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.data.DLReadableBuffer;
import org.knime.dl.core.data.DLWritableBuffer;
import org.knime.dl.core.execution.DLAbstractNetworkExecutionSession;
import org.knime.dl.core.execution.DLExecutionMonitor;
import org.knime.dl.core.execution.DLExecutionStatus;
import org.knime.dl.core.execution.DLNetworkOutputConsumer;
import org.knime.dl.tensorflow.core.execution.TFNetworkExecutionSession;
import org.knime.dl.tensorflow.savedmodel.core.TFSavedModelNetwork;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorReadableBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorWritableBuffer;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session.Runner;
import org.tensorflow.Tensor;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public class TFSavedModelNetworkExecutionSession extends
	DLAbstractNetworkExecutionSession<TFSavedModelNetwork> implements TFNetworkExecutionSession {

	private SavedModelBundle m_savedModelBundle;

	/**
	 * Creates a new execution session for a TensorFlow SavedModel deep learning network.
	 *
	 * @param network the deep learning network
	 * @param executionInputSpecs the tensor spec of the inputs
	 * @param requestedOutputs the ids of the requested outputs
	 * @param inputPreparer an input preparer
	 * @param outputConsumer an output consumer
	 * @param tensorFactory a tensor factory which creates tensors with {@link TFTensorReadableBuffer}s and
	 *            {@link TFTensorWritableBuffer}s.
	 */
	public TFSavedModelNetworkExecutionSession(final TFSavedModelNetwork network,
			final Set<DLTensorSpec> executionInputSpecs, final Set<DLTensorId> requestedOutputs,
			final DLNetworkInputPreparer inputPreparer, final DLNetworkOutputConsumer outputConsumer,
			final DLTensorFactory tensorFactory) {
		super(network, executionInputSpecs, requestedOutputs, inputPreparer, outputConsumer, tensorFactory);
	}

	@Override
	protected void executeInternal(final DLExecutionMonitor monitor) throws DLCanceledExecutionException, Exception {
		if (m_savedModelBundle == null) {
			m_savedModelBundle = SavedModelBundle.load(m_network.getSavedModelInDir().getAbsolutePath(),
					m_network.getSpec().getTags());
		}

		final DLExecutionStatus status = monitor.getExecutionStatus();
		final long numBatches = m_inputPreparer.getNumBatches();

		// Loop over batches
		for (long i = 0; i < numBatches; i++) {
			// Create a TensorFlow runner
			try (final DLRunner runner = new DLRunner(m_savedModelBundle.session().runner())) {
				monitor.checkCanceled();

				// Prepare the inputs
				m_inputPreparer.prepare(m_input, i);
				monitor.checkCanceled();

				// Feed the inputs
				m_input.entrySet().stream().forEach(e -> runner.feed(e.getKey(), e.getValue()));
				monitor.checkCanceled();

				// Request the outputs
				m_requestedOutputs.stream().forEach(id -> runner.fetch(id));
				monitor.checkCanceled();

				// Run the model
				runner.run();
				monitor.checkCanceled();

				// Reset the buffers of the input tensors
				m_input.values().forEach(in -> in.getBuffer().reset());

				// Create the output map if it doesn't exist yet
				if (m_output == null) {
					m_output = new HashMap<>(m_requestedOutputs.size());

					// Fill a Map with the specs for a tensor id
					final Map<DLTensorId, DLTensorSpec> allOutputSpecs = new HashMap<>(m_requestedOutputs.size());
					Arrays.stream(ArrayUtils.addAll(m_network.getSpec().getOutputSpecs(),
							m_network.getSpec().getHiddenOutputSpecs()))
							.filter(s -> m_requestedOutputs.contains(s.getIdentifier()))
							.forEach(s -> allOutputSpecs.put(s.getIdentifier(), s));

					// Create tensors for the requested outputs
					for (final DLTensorId id : m_requestedOutputs) {
						final long[] outShape = runner.getOutputShape(id);
						final long outBatchSize = outShape[0];
						final long[] outShapeWithoutBatchSize = Arrays.stream(outShape).skip(1).toArray();
						final DLTensorSpec executionSpec = m_tensorFactory.createExecutionTensorSpec(
								allOutputSpecs.get(id), outBatchSize, outShapeWithoutBatchSize);
						m_output.put(id, m_tensorFactory.createReadableTensor(executionSpec));
					}
					monitor.checkCanceled();
				}

				// Fill the output tensors
				m_output.entrySet().forEach(e -> runner.fillTensor(e.getKey(), e.getValue()));
				monitor.checkCanceled();

				// Consume the output
				m_outputConsumer.accept(m_output);

				// Reset the buffers of the output tensors
				m_output.values().stream().forEach(o -> o.getBuffer().reset());

				// This batch is done!
				status.batchEnded().raise(null);
			}
		}
	}

	@Override
	public void close() throws Exception {
		if (m_savedModelBundle != null) {
			m_savedModelBundle.close();
		}
		super.close();
	}

	private static Tensor<?> createTFTensor(final DLTensor<? extends DLWritableBuffer> dlTensor) {
		final DLFixedTensorShape shape;
		try {
			shape = (DLFixedTensorShape) dlTensor.getSpec().getShape();
		} catch (final ClassCastException e) {
			throw new IllegalStateException("The shape of the tensor must be known at runtime", e);
		}

		final long batchSize = dlTensor.getBuffer().size() / dlTensor.getExampleSize();

		try {
			final TFTensorWritableBuffer<?> buffer = (TFTensorWritableBuffer<?>) dlTensor
					.getBuffer();
			return buffer.readIntoTensor(batchSize, shape);
		} catch (final ClassCastException e) {
			throw new IllegalStateException("Wrong type of buffer: \"" + dlTensor.getBuffer().getClass()
					+ "\", expected: \"" + TFTensorWritableBuffer.class + "\".");
		}
	}

	private static class DLRunner implements AutoCloseable {

		private final Runner m_runner;

		/** Keep track of open Tensors to close them */
		private final List<Tensor<?>> m_openTensors = new ArrayList<>();

		private final List<DLTensorId> m_outputIds = new ArrayList<>();

		private List<Tensor<?>> m_outputs;

		public DLRunner(final Runner runner) {
			m_runner = runner;
		}

		public void feed(final DLTensorId id, final DLTensor<? extends DLWritableBuffer> tensor) {
			Tensor<?> t = createTFTensor(tensor);
			m_openTensors.add(t);
			m_runner.feed(id.getIdentifierString(), t);
		}

		public void fetch(final DLTensorId id) {
			m_runner.fetch(id.getIdentifierString(), m_outputIds.size());
			m_outputIds.add(id);
		}

		public void run() {
			m_outputs = m_runner.run();
		}

		public long[] getOutputShape(final DLTensorId id) {
			return m_outputs.get(m_outputIds.indexOf(id)).shape();
		}

		public void fillTensor(final DLTensorId id, final DLTensor<? extends DLReadableBuffer> tensor) {
			final TFTensorReadableBuffer buffer;
			try {
				buffer = (TFTensorReadableBuffer) tensor.getBuffer();
			} catch (final ClassCastException e) {
				throw new IllegalStateException("Wrong type of buffer: \"" + tensor.getBuffer().getClass()
						+ "\", expected: \"" + TFTensorReadableBuffer.class + "\".");
			}
			buffer.writeFromTensor(m_outputs.get(m_outputIds.indexOf(id)));
		}

		@Override
		public void close() throws IOException {
			m_openTensors.forEach(Tensor::close);
			m_openTensors.clear();
			m_outputs.forEach(Tensor::close);
			m_outputs.clear();
		}
	}
}
