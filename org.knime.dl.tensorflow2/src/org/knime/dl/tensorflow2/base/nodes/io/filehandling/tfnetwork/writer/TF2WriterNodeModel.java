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
 *   May 26, 2020 (benjamin): created
 */
package org.knime.dl.tensorflow2.base.nodes.io.filehandling.tfnetwork.writer;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Optional;
import java.util.zip.Deflater;

import org.apache.commons.io.FilenameUtils;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.context.NodeCreationConfiguration;
import org.knime.core.node.port.PortObject;
import org.knime.core.util.PathUtils;
import org.knime.dl.core.DLExecutionMonitorCancelable;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonNetworkHandle;
import org.knime.dl.python.core.DLPythonNetworkLoaderRegistry;
import org.knime.dl.python.util.DLPythonUtils;
import org.knime.dl.tensorflow2.base.portobjects.TF2NetworkPortObject;
import org.knime.dl.tensorflow2.core.TF2Network;
import org.knime.dl.tensorflow2.core.TF2NetworkLoader;
import org.knime.dl.tensorflow2.core.TF2PythonContext;
import org.knime.filehandling.core.defaultnodesettings.filechooser.writer.FileOverwritePolicy;
import org.knime.filehandling.core.node.portobject.writer.PortObjectToPathWriterNodeModel;

/**
 * Node model of the TensorFlow writer node.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
final class TF2WriterNodeModel extends PortObjectToPathWriterNodeModel<TF2WriterNodeConfig> {

    enum NetworkFormat {
            SAVED_MODEL("tf", ""), SAVED_MODEL_ZIP("tf", "zip"), H5("h5", "h5");

        private final String m_tfSaveFormat;

        private final String m_fileExtension;

        private NetworkFormat(final String tfSaveFormat, final String fileExtension) {
            m_tfSaveFormat = tfSaveFormat;
            m_fileExtension = fileExtension;
        }
    }

    protected TF2WriterNodeModel(final NodeCreationConfiguration creationConfig, final TF2WriterNodeConfig config) {
        super(creationConfig, config);
    }

    @Override
    protected void writeToPath(final PortObject object, final Path outputPath, final ExecutionContext exec)
        throws Exception {
        final TF2Network network = ((TF2NetworkPortObject)object).getNetwork();
        final DLExecutionMonitorCancelable cancelable = new DLExecutionMonitorCancelable(exec);
        final NetworkFormat format = getNetworkFormat(outputPath);
        final boolean saveOptimizerState = getConfig().getSaveOptimizerStateModel().getBooleanValue();
        final boolean writeDirectly = isLocalPath(outputPath) && !NetworkFormat.SAVED_MODEL_ZIP.equals(format);

        // Fail fast if overwrite is disabled but the output exists
        checkOverwrite(outputPath);

        // Get the path for writing the model to using Python
        final Path modelPath;
        if (writeDirectly) {
            modelPath = outputPath;
        } else if (NetworkFormat.H5.equals(format)) {
            modelPath = PathUtils.createTempFile("TF2_network", "h5");
        } else {
            modelPath = PathUtils.createTempDir("TF2_network").resolve(getModelName(outputPath));
        }

        // Save the model to the model path (can be a temporary directory)
        final TF2NetworkLoader loader = new TF2NetworkLoader();
        loader.checkAvailability(false, DLPythonNetworkLoaderRegistry.getInstallationTestTimeout(), cancelable);
        try (final DLPythonContext context = new TF2PythonContext()) {
            final DLPythonNetworkHandle handle = loader.load(network, context, saveOptimizerState, cancelable);
            final String saveCode = getSaveNetworkCode(handle, saveOptimizerState, modelPath.toString(), format);
            context.executeInKernel(saveCode, cancelable);
        }

        // Copy/ZIP the file to the output
        if (!writeDirectly) {
            switch (format) {
                case SAVED_MODEL:
                    PathUtils.copyDirectory(modelPath, outputPath);
                    break;

                case SAVED_MODEL_ZIP:
                    PathUtils.zip(modelPath, openOutputStream(outputPath), Deflater.DEFAULT_COMPRESSION);
                    break;

                case H5:
                    Files.copy(modelPath, openOutputStream(outputPath));
                    break;

                default:
                    // Cannot happen
                    throw new IllegalStateException(
                        "The format \"" + format + "\" is not supported. This is an implementation error.");
            }

            // Delete the temp file
            PathUtils.deleteDirectoryIfExists(modelPath);
        }
    }

    /** Open an output stream at the given location with the configured open options */
    private OutputStream openOutputStream(final Path outputPath) throws IOException {
        final OpenOption[] openOptions =
            getConfig().getFileChooserModel().getFileOverwritePolicy().getOpenOptions();
        return Files.newOutputStream(outputPath, openOptions);
    }

    /** @throws FileAlreadyExistsException if the given file exits and shoul not be overwritten */
    private void checkOverwrite(final Path outputPath) throws FileAlreadyExistsException {
        if (getConfig().getFileChooserModel().getFileOverwritePolicy() == FileOverwritePolicy.FAIL &&
                Files.exists(outputPath)) {
            throw new FileAlreadyExistsException(outputPath.toString());
        }
    }

    private static String getModelName(final Path outputPath) {
        return FilenameUtils.getBaseName(outputPath.toString());
    }

    /** Determine the format based on the file extension of the path */
    private static NetworkFormat getNetworkFormat(final Path outputPath) throws InvalidSettingsException {
        final String fileExtension = FilenameUtils.getExtension(outputPath.toString());
        final Optional<NetworkFormat> format =
            Arrays.stream(NetworkFormat.values()).filter(f -> f.m_fileExtension.equals(fileExtension)).findFirst();
        // If not zip or h5 we save a SavedModel
        return format.orElse(NetworkFormat.SAVED_MODEL);
    }

    /** Determine if the given path is local and can be used in the Python snippet */
    private static boolean isLocalPath(final Path outputPath) {
        // Not sure if this always works
        // However, if the file is local and the check fails the node will still work
        // by using a temporary directory
        return outputPath.toUri().getScheme().equals("file");
    }

    /** Create code for saving the network in the given format */
    private static String getSaveNetworkCode(final DLPythonNetworkHandle network, final boolean saveOptimizerState,
        final String path, final NetworkFormat format) {
        return DLPythonUtils.createSourceCodeBuilder() //
            .a("import DLPythonNetwork") //
            .n("network = DLPythonNetwork.get_network(").as(network.getIdentifier()).a(")") //
            .n("network.model.save(") //
            /**/ .asr(path).a(",") //
            /**/ .a("include_optimizer=").a(saveOptimizerState).a(",") //
            /**/ .a("save_format=").as(format.m_tfSaveFormat) //
            /**/ .a(")") //
            .toString();
    }
}
