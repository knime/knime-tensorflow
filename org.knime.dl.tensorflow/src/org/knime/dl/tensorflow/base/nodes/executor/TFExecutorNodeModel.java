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
package org.knime.dl.tensorflow.base.nodes.executor;

import java.util.Map;

import org.knime.core.data.DataRow;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeLogger;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.dl.core.DLExecutionSpecCreator;
import org.knime.dl.core.DLMissingExtensionException;
import org.knime.dl.core.DLNetwork;
import org.knime.dl.core.DLNetworkInputPreparer;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.data.convert.DLTensorToDataCellConverterFactory;
import org.knime.dl.core.execution.DLExecutionContext;
import org.knime.dl.core.execution.DLNetworkExecutionSession;
import org.knime.dl.core.execution.DLNetworkOutputConsumer;
import org.knime.dl.python.base.node.DLAbstractPythonBasedExecutorNodeModel;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.prefs.DLPythonPreferences;
import org.knime.dl.tensorflow.base.nodes.TFConfigProtoConfig;
import org.knime.dl.tensorflow.base.portobjects.TFNetworkPortObject;
import org.knime.dl.tensorflow.savedmodel.core.execution.TFPythonSavedModelExecutionContext;
import org.knime.dl.tensorflow.savedmodel.core.execution.TFSavedModelNetworkExecutionSession;
import org.knime.python2.PythonCommand;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFExecutorNodeModel extends DLAbstractPythonBasedExecutorNodeModel {

	private static final NodeLogger LOGGER = NodeLogger.getLogger(TFExecutorNodeModel.class);

	static PythonCommand getDefaultPythonCommand() {
	    return DLPythonPreferences.getPythonCommandPreference();
	}

	static TFConfigProtoConfig createConfigProtoConfig() {
		return new TFConfigProtoConfig();
	}

	private final TFConfigProtoConfig m_configProtoConfig;

	TFExecutorNodeModel() {
		super(TFNetworkPortObject.TYPE, TFExecutorNodeModel::getDefaultPythonCommand);
		m_configProtoConfig = createConfigProtoConfig();
	}

    @Override
    protected DLPythonContext getContext(final DLExecutionContext<?, ?> ctx) {
        if (ctx instanceof TFPythonSavedModelExecutionContext) {
            return super.getContext(ctx);
        } else {
            return null;
        }
    }

    /**
     * Note: {@code context} will be {@code null} if the Java back end is selected (see above).
     */
    @Override
    protected <N extends DLNetwork> DLNetworkExecutionSession createExecutionSession(final DLPythonContext context,
        final N network, final int batchSize, final Map<DLTensorId, int[]> columnsForTensorId,
        final Map<DLTensorId, DLTensorToDataCellConverterFactory<?, ?>> outputConverterForTensorId,
        final DataRow firstRow, final DLNetworkInputPreparer inputPreparer,
        final DLNetworkOutputConsumer outputConsumer) throws DLMissingExtensionException, InvalidSettingsException {

		final DLExecutionContext<DLPythonContext, N> ctx = getExecutionContext(context);
		final DLNetworkExecutionSession session = ctx.createExecutionSession(context, network,
				DLExecutionSpecCreator.createExecutionSpecs(firstRow, ctx.getTensorFactory(), batchSize,
						columnsForTensorId, m_inputConverters),
				outputConverterForTensorId.keySet(), inputPreparer, outputConsumer);

		final String visibleDeviceList = m_configProtoConfig.getVisibleDevicesList().getValue();
		final double perProcessGpuMem = m_configProtoConfig.getPerProcessGpuMem().getValue();
		if (session instanceof TFSavedModelNetworkExecutionSession) {
			final TFSavedModelNetworkExecutionSession s = (TFSavedModelNetworkExecutionSession) session;
			s.setVisibleDeviceList(visibleDeviceList);
			s.setPerProcessGpuMemoryFraction(perProcessGpuMem);
		} else {
			if (!visibleDeviceList.equals(TFConfigProtoConfig.VISIBLE_DEVICES_LIST_DEFAULT)) {
				LOGGER.warn("Could not apply the value of 'Visible devices list' because the selected backend "
						+ "does not support setting this value.");
			}
			if (perProcessGpuMem != TFConfigProtoConfig.PER_PROCESS_GPU_MEM_DEFAULT) {
				LOGGER.warn("Could not apply the value of 'Per process GPU memory fraction' because "
						+ "the selected backend does not support setting this value.");
			}
		}
		return session;
	}

	@Override
	protected void saveSettingsTo(final NodeSettingsWO settings) {
		super.saveSettingsTo(settings);
		try {
			m_configProtoConfig.saveToSettings(settings);
		} catch (final InvalidSettingsException e) {
			throw new RuntimeException(e.getMessage(), e);
		}
	}

	@Override
	protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
		super.loadValidatedSettingsFrom(settings);
		m_configProtoConfig.loadFromSettings(settings);
	}
}
