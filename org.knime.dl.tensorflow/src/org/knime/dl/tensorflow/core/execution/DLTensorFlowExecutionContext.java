package org.knime.dl.tensorflow.core.execution;

import java.util.Set;

import org.knime.dl.core.DLNetworkInputPreparer;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.execution.DLExecutionContext;
import org.knime.dl.core.execution.DLNetworkOutputConsumer;
import org.knime.dl.tensorflow.core.DLTensorFlowNetwork;

/**
 * A execution context for TensorFlow deep learning networks.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
interface DLTensorFlowExecutionContext<N extends DLTensorFlowNetwork> extends DLExecutionContext<N> {

	@Override
	DLTensorFlowNetworkExecutionSession createExecutionSession(N network, Set<DLTensorSpec> executionInputSpecs,
			Set<DLTensorId> requestedOutputs, DLNetworkInputPreparer inputPreparer,
			DLNetworkOutputConsumer outputConsumer);
}
