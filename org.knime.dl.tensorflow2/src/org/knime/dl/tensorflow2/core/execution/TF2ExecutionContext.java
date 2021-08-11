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
package org.knime.dl.tensorflow2.core.execution;

import java.util.Set;

import org.knime.dl.core.DLCancelable;
import org.knime.dl.core.DLCanceledExecutionException;
import org.knime.dl.core.DLInstallationTestTimeoutException;
import org.knime.dl.core.DLMissingDependencyException;
import org.knime.dl.core.DLNetworkInputPreparer;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.execution.DLExecutionContext;
import org.knime.dl.core.execution.DLNetworkOutputConsumer;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonDefaultTensorFactory;
import org.knime.dl.python.prefs.DLPythonPreferences;
import org.knime.dl.tensorflow2.core.TF2Network;
import org.knime.dl.tensorflow2.core.TF2NetworkLoader;
import org.knime.dl.tensorflow2.core.TF2PythonContext;

/**
 * The execution context for a {@link TF2Network} using the Python API.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TF2ExecutionContext implements DLExecutionContext<DLPythonContext, TF2Network> {

    private static final String EXECUTION_CONTEXT_NAME = "TensorFlow 2 (Python)";

    @Override
    public Class<TF2Network> getNetworkType() {
        return TF2Network.class;
    }

    @Override
    public String getName() {
        return EXECUTION_CONTEXT_NAME;
    }

    @Override
    public DLPythonDefaultTensorFactory getTensorFactory() {
        return new DLPythonDefaultTensorFactory();
    }

    @Deprecated
    @Override
    public DLPythonContext createDefaultContext() {
        return new TF2PythonContext(DLPythonPreferences.getPythonTF2CommandPreference());
    }

    @Override
    public void checkAvailability(final DLPythonContext context, final boolean forceRefresh, final int timeout,
        final DLCancelable cancelable)
        throws DLMissingDependencyException, DLInstallationTestTimeoutException, DLCanceledExecutionException {
        new TF2NetworkLoader().checkAvailability(context, forceRefresh, timeout, cancelable);
    }

    @Override
    public TF2ExecutionSession createExecutionSession(final DLPythonContext context, final TF2Network network,
        final Set<DLTensorSpec> executionInputSpecs, final Set<DLTensorId> requestedOutputs,
        final DLNetworkInputPreparer inputPreparer, final DLNetworkOutputConsumer outputConsumer) {
        return new TF2ExecutionSession(context, network, executionInputSpecs, requestedOutputs, inputPreparer,
            outputConsumer, getTensorFactory());
    }
}
