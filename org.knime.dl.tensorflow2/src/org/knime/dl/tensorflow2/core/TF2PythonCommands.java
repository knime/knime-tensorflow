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
 * History
 *   Apr 27, 2020 (benjamin): created
 */
package org.knime.dl.tensorflow2.core;

import java.io.File;
import java.io.IOException;

import org.knime.core.node.NodeLogger;
import org.knime.core.util.Version;
import org.knime.dl.core.DLCancelable;
import org.knime.dl.core.DLCanceledExecutionException;
import org.knime.dl.core.DLInvalidEnvironmentException;
import org.knime.dl.core.DLNetworkInputProvider;
import org.knime.dl.core.DLTensor;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.data.DLWritableBuffer;
import org.knime.dl.core.training.DLTrainingMonitor;
import org.knime.dl.python.core.DLPythonAbstractCommands;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonNetworkHandle;
import org.knime.dl.python.core.SingleValueTableCreator;
import org.knime.dl.python.core.training.DLPythonTrainingStatus;
import org.knime.dl.python.util.DLPythonSourceCodeBuilder;
import org.knime.dl.python.util.DLPythonUtils;
import org.knime.dl.util.DLThrowingLambdas.DLThrowingBiFunction;
import org.knime.dl.util.DLUtils;
import org.knime.python2.extensions.serializationlibrary.interfaces.Cell;
import org.knime.python2.extensions.serializationlibrary.interfaces.TableChunker;

/**
 * Python commands for a {@link TF2Network}.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public final class TF2PythonCommands extends DLPythonAbstractCommands {

    private static final NodeLogger LOGGER = NodeLogger.getLogger(TF2PythonCommands.class);

    private static final String TF_VERSION_NAME = "tf_version";

    private static final Version MIN_TF_VERSION = new Version(2, 2, 0);

    /**
     * Create Python commands for handling TensorFlow 2 networks. A new Python context to communicate with Python is
     * created automatically and has to be closed by calling the {@link #close()} method of this object.
     */
    @SuppressWarnings("resource") // Context is closed in #close
    public TF2PythonCommands() {
        super(new TF2PythonContext());
    }

    /**
     * Create Python commands for handling TensorFlow 2 networks.
     *
     * @param context The context which is used to communicate with Python. Note that this context is closed when
     *            {@link #close()} is called.
     */
    public TF2PythonCommands(final DLPythonContext context) {
        super(context);
    }

    @Override
    @SuppressWarnings("resource") // Context is closed in close method
    public TF2NetworkSpec extractNetworkSpec(final DLPythonNetworkHandle network, final DLCancelable cancelable)
        throws DLInvalidEnvironmentException, IOException, DLCanceledExecutionException {
        // Input, hidden, output specs
        getContext(cancelable).executeInKernel(getExtractNetworkSpecsCode(network), cancelable);
        final DLTensorSpec[] inputSpecs = extractTensorSpec(INPUT_SPECS_NAME, cancelable);
        final DLTensorSpec[] hiddenOutputSpecs = extractTensorSpec(HIDDEN_OUTPUT_SPECS_NAME, cancelable);
        final DLTensorSpec[] outputSpecs = extractTensorSpec(OUTPUT_SPECS_NAME, cancelable);

        // Package versions
        final Version pythonVersion = getPythonVersion(cancelable);
        final Version tfVersion = getTFVersion(cancelable);

        if (tfVersion.compareTo(MIN_TF_VERSION) < 0) {
            throw new DLInvalidEnvironmentException(
                "TensorFlow 2.2.0 or higher is required to use TensorFlow Keras models. The installed TensorFlow version is \""
                    + tfVersion.toString() + "\". Please update TensorFlow.");

        }

        return new TF2NetworkSpec(pythonVersion, tfVersion, inputSpecs, hiddenOutputSpecs, outputSpecs);
    }

    @Override
    public DLPythonNetworkHandle loadNetwork(final String path, final boolean loadTrainingConfig,
        final DLCancelable cancelable) throws DLInvalidEnvironmentException, IOException, DLCanceledExecutionException {
        try {
            // Try to load it as requested
            return super.loadNetwork(path, loadTrainingConfig, cancelable);
        } catch (final IOException e) {
            if (loadTrainingConfig) {
                // If we tried to load it compiled this could have been the problem
                // -> Warn the user and try it uncompiled
                LOGGER.warn("Couldn't load the model with the training configuration. See log for details. "
                    + "Trying to load it uncompiled.", e);
                return super.loadNetwork(path, false, cancelable);
            }
            throw e;
        }
    }

    @Override
    protected String getSetupEnvironmentCode() {
        // Nothing to do
        return "";
    }

    @Override
    protected File getInstallationTestFile() throws IOException {
        return DLUtils.Files.getFileFromSameBundle(this, "py/TF2NetworkTester.py");
    }

    @Override
    protected String getSetupBackendCode() {
        // Nothing to do
        return "";
    }

    @Override
    protected TF2NetworkReaderCommands getNetworkReaderCommands() {
        return new TF2NetworkReaderCommands();
    }

    @Override
    protected DLPythonNetworkTrainingTaskHandler createNetworkTrainingTaskHandler(final DLPythonContext context,
        final DLTrainingMonitor<? extends DLPythonTrainingStatus> monitor,
        final DLNetworkInputProvider trainingInputProvider, final DLNetworkInputProvider validationInputProvider,
        final DLThrowingBiFunction<DLTensorId, DLTensor<? extends DLWritableBuffer>, TableChunker, IOException> singleTensorTableChunkerCreator) {
        // TODO implement
        throw new UnsupportedOperationException("Not yet implemented");
    }

    /**
     * @param cancelable to check if the execution has been canceled
     * @return the TensorFlow version
     * @throws DLCanceledExecutionException if the execution has been canceled
     * @throws DLInvalidEnvironmentException if failed to properly setup the Python context
     * @throws IOException if getting the data from python failed
     */
    @SuppressWarnings("resource") // Context is closed in close method
    private Version getTFVersion(final DLCancelable cancelable)
        throws DLCanceledExecutionException, DLInvalidEnvironmentException, IOException {
        final DLPythonSourceCodeBuilder b = DLPythonUtils.createSourceCodeBuilder() //
            .a("import tensorflow as tf") //
            .n("import pandas as pd") //
            .n("global ").a(TF_VERSION_NAME) //
            .n(TF_VERSION_NAME).a(" = pd.DataFrame([tf.__version__])"); //
        getContext(cancelable).executeInKernel(b.toString(), cancelable);
        final String kerasVersion = (String)getContext(cancelable).getDataFromKernel(TF_VERSION_NAME,
            (s, ts) -> new SingleValueTableCreator<>(s, Cell::getStringValue), cancelable).getTable();
        return new Version(kerasVersion);
    }

    private static class TF2NetworkReaderCommands extends DLPythonAbstractNetworkReaderCommands {

        protected TF2NetworkReaderCommands() {
            super("from TF2Network import TF2NetworkReader", "TF2NetworkReader()");
        }

        @Override
        public String read(final String path, final boolean loadTrainingConfig) {
            return DLPythonUtils.createSourceCodeBuilder() //
                .a("read(").asr(path).a(", compile=").a(loadTrainingConfig).a(")") //
                .toString();
        }
    }
}
