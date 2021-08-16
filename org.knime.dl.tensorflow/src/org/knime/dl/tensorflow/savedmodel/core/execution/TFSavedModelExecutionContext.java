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

import java.util.Set;

import org.knime.dl.core.DLCancelable;
import org.knime.dl.core.DLCanceledExecutionException;
import org.knime.dl.core.DLInstallationTestTimeoutException;
import org.knime.dl.core.DLMissingDependencyException;
import org.knime.dl.core.DLNetworkInputPreparer;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.execution.DLNetworkOutputConsumer;
import org.knime.dl.tensorflow.core.execution.TFAbstractExecutionContext;
import org.knime.dl.tensorflow.core.execution.TFNetworkExecutionSession;
import org.knime.dl.tensorflow.savedmodel.core.TFSavedModelNetwork;
import org.knime.dl.tensorflow.savedmodel.core.TFSavedModelTensorFactory;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFSavedModelExecutionContext extends TFAbstractExecutionContext<Void, TFSavedModelNetwork> {

	private static final String EXECUTION_CONTEXT_NAME = "TensorFlow";

	/**
	 * Creates a new execution context for TensorFlow SavedModels.
	 */
	public TFSavedModelExecutionContext() {
		super(TFSavedModelNetwork.class, new TFSavedModelTensorFactory(), EXECUTION_CONTEXT_NAME);
	}

    @Deprecated
    @Override
    public Void createDefaultContext() {
        // No external context.
        return null;
    }

	@Override
    public void checkAvailability(final Void noContext, final boolean forceRefresh, final int timeout,
        final DLCancelable cancelable)
        throws DLMissingDependencyException, DLInstallationTestTimeoutException, DLCanceledExecutionException {
		// No-op: No external dependencies.
	}

    @Override
    public TFNetworkExecutionSession createExecutionSession(final Void noContext, final TFSavedModelNetwork network,
        final Set<DLTensorSpec> executionInputSpecs, final Set<DLTensorId> requestedOutputs,
        final DLNetworkInputPreparer inputPreparer, final DLNetworkOutputConsumer outputConsumer) {
        return new TFSavedModelNetworkExecutionSession(network, executionInputSpecs, requestedOutputs, inputPreparer,
            outputConsumer, getTensorFactory());
    }
}
