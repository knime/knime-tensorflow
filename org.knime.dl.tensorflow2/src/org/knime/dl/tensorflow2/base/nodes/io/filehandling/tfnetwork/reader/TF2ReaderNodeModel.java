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
 *   May 22, 2020 (benjamin): created
 */
package org.knime.dl.tensorflow2.base.nodes.io.filehandling.tfnetwork.reader;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Iterator;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

import org.apache.commons.io.IOUtils;
import org.knime.core.data.filestore.FileStore;
import org.knime.core.node.ExecutionContext;
import org.knime.core.node.InvalidSettingsException;
import org.knime.core.node.NodeSettingsRO;
import org.knime.core.node.NodeSettingsWO;
import org.knime.core.node.context.NodeCreationConfiguration;
import org.knime.core.node.port.PortObject;
import org.knime.core.util.PathUtils;
import org.knime.dl.base.portobjects.DLNetworkPortObject;
import org.knime.dl.core.DLExecutionMonitorCancelable;
import org.knime.dl.core.DLNetworkFileStoreLocation;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonNetworkHandle;
import org.knime.dl.python.core.DLPythonNetworkLoaderRegistry;
import org.knime.dl.python.prefs.DLPythonPreferences;
import org.knime.dl.tensorflow2.base.portobjects.TF2NetworkPortObject;
import org.knime.dl.tensorflow2.core.TF2Network;
import org.knime.dl.tensorflow2.core.TF2NetworkLoader;
import org.knime.dl.tensorflow2.core.TF2PythonContext;
import org.knime.filehandling.core.node.portobject.reader.PortObjectFromPathReaderNodeModel;
import org.knime.filehandling.core.node.portobject.reader.PortObjectReaderNodeConfig;
import org.knime.python2.PythonVersion;
import org.knime.python2.config.PythonCommandConfig;

/**
 * Node model of the TensorFlow 2 network reader node.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
final class TF2ReaderNodeModel extends PortObjectFromPathReaderNodeModel<PortObjectReaderNodeConfig> {

    private enum NetworkFormat {
            SAVED_MODEL, SAVED_MODEL_ZIP, H5
    }

    private static final String SAVED_MODEL_REGEX =
        "^.*saved_model.pb$" + "|^.*variables(/.*|\\.*)?$" + "|^.*assets(/.*|\\.*)?$";

    static PythonCommandConfig createPythonCommandConfig() {
        return new PythonCommandConfig(PythonVersion.PYTHON3, DLPythonPreferences::getCondaInstallationPath,
            DLPythonPreferences::getPythonTF2CommandPreference);
    }

    private final PythonCommandConfig m_pythonCommandConfig = createPythonCommandConfig();

    TF2ReaderNodeModel(final NodeCreationConfiguration creationConfig, final PortObjectReaderNodeConfig config) {
        super(creationConfig, config);
    }

    @Override
    protected void saveSettingsTo(final NodeSettingsWO settings) {
        super.saveSettingsTo(settings);
        m_pythonCommandConfig.saveSettingsTo(settings);
    }

    @Override
    protected void validateSettings(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_pythonCommandConfig.loadSettingsFrom(settings);
        super.validateSettings(settings);
    }

    @Override
    protected void loadValidatedSettingsFrom(final NodeSettingsRO settings) throws InvalidSettingsException {
        m_pythonCommandConfig.loadSettingsFrom(settings);
        super.loadValidatedSettingsFrom(settings);
    }

    @Override
    protected PortObject[] readFromPath(final Path inputPath, final ExecutionContext exec) throws Exception {
        final DLExecutionMonitorCancelable cancelable = new DLExecutionMonitorCancelable(exec);
        final NetworkFormat format = getFormat(inputPath);

        // File store for the final model
        final FileStore fileStore = DLNetworkPortObject.createFileStoreForSaving("", exec);
        final Path fileStorePath = fileStore.getFile().toPath();
        final URI fileStoreUri = fileStorePath.toUri();

        // A temporary file for non-local h5 files
        Path tmpDir = null;

        // Get the network location:
        // SavedModel: Copy into file store
        // SavedModel ZIP: Extract into file store
        // H5: Keep at location or copy to tmp file. Will be saved to file store later
        final URI networkUri;
        switch (format) {
            case SAVED_MODEL:
                // Copy the SavedModel into the file store
                PathUtils.copyDirectory(inputPath, fileStorePath);
                networkUri = fileStoreUri;
                break;

            case SAVED_MODEL_ZIP:
                // Unzip the SavedModel into the file store
                extractZipSavedModel(inputPath, fileStorePath);
                networkUri = fileStoreUri;
                break;

            case H5:
                if (isLocalPath(inputPath)) {
                    networkUri = inputPath.toUri();
                } else {
                    tmpDir = PathUtils.createTempDir("tf_network");
                    final Path tmpFile = tmpDir.resolve("network.h5");
                    Files.copy(inputPath, tmpFile);
                    networkUri = tmpFile.toUri();
                }
                break;

            default:
                // Cannot happen
                throw new IllegalStateException(
                    "The format \"" + format + "\" is not supported. This is an implementation error.");
        }

        // Load the model
        final TF2Network network;
        try (final DLPythonContext context = new TF2PythonContext(m_pythonCommandConfig.getCommand())) {
            final TF2NetworkLoader loader = new TF2NetworkLoader();
            loader.checkAvailability(context, false, DLPythonNetworkLoaderRegistry.getInstallationTestTimeout(),
                cancelable);
            final DLPythonNetworkHandle handle = loader.load(networkUri, context, true, cancelable);

            // For H5 the network is not in the file store yet. Save it to the file store
            if (NetworkFormat.H5 == format) {
                loader.save(handle, fileStoreUri, context, cancelable);
            }

            network = loader.fetch(handle, new DLNetworkFileStoreLocation(fileStore), context, cancelable);
        }

        // Delete the temporary file if it exists
        if (tmpDir != null) {
            PathUtils.deleteDirectoryIfExists(tmpDir);
        }

        // Create the port object
        final TF2NetworkPortObject portObject = new TF2NetworkPortObject(network, fileStore);
        return new PortObject[]{portObject};
    }

    /** Get the format of the given path
     * @throws IOException */
    private static NetworkFormat getFormat(final Path path) throws InvalidSettingsException, IOException {
        final BasicFileAttributes attr = Files.readAttributes(path, BasicFileAttributes.class);
        if (attr.isRegularFile() && path.toString().endsWith(".zip")) {
            return NetworkFormat.SAVED_MODEL_ZIP;
        } else if (attr.isRegularFile() && path.toString().endsWith(".h5")) {
            return NetworkFormat.H5;
        } else if (attr.isDirectory()) {
            return NetworkFormat.SAVED_MODEL;
        } else {
            throw new InvalidSettingsException(
                "The selected path is not supported. Must be a SavedModel directory, ZIP file or H5 file.");
        }
    }

    /** Extract the ZIP file to a file store with the SavedModel at the root */
    private static void extractZipSavedModel(final Path zipFile, final Path target) throws IOException {
        Files.createDirectories(target);

        // The prefix from the root of the zip file to the SavedModel
        String prefix = "";

        // Extract the zip file
        try (final InputStream inputStream = Files.newInputStream(zipFile, StandardOpenOption.READ);
                final ZipInputStream zipStream = new ZipInputStream(inputStream)) {
            ZipEntry entry;
            while ((entry = zipStream.getNextEntry()) != null) {
                // Check if the entry is a relevant file in the SavedModel we
                // extract
                final String name = entry.getName();
                if (!name.matches(SAVED_MODEL_REGEX) || !name.startsWith(prefix)) {
                    continue;
                }

                // If this is the saved_model.pb we use it to determine the prefix
                if (prefix.isEmpty() && name.matches("^.*saved_model.pb?$")) {
                    prefix = name.substring(0, name.lastIndexOf("saved_model.pb"));
                }

                // Extract this entry
                final Path destPath = target.resolve(name);
                if (entry.isDirectory()) {
                    Files.createDirectories(destPath);
                } else {
                    Files.createDirectories(destPath.getParent());
                    try (final OutputStream out = Files.newOutputStream(destPath)) {
                        IOUtils.copy(zipStream, out);
                    }
                }
            }
        }

        // Move the SavedModel to the root of the FileStore
        final Path savedModelPath = target.resolve(prefix);
        if (!savedModelPath.equals(target)) {
            try (Stream<Path> savedModelFilesStream = Files.list(savedModelPath)) {
                final Iterator<Path> savedModelFiles = savedModelFilesStream.iterator();
                while (savedModelFiles.hasNext()) {
                    final Path f = savedModelFiles.next();
                    Files.move(f, target.resolve(f.getFileName()));
                }
            }

            // Delete everything else
            try (Stream<Path> pathsToDeleteStream = Files.list(target)) {
                final Iterator<Path> pathsToDelete = pathsToDeleteStream //
                        .filter(p -> !p.getFileName().toString().matches(SAVED_MODEL_REGEX)).iterator();
                while (pathsToDelete.hasNext()) {
                    PathUtils.deleteDirectoryIfExists(pathsToDelete.next());
                }
            }
        }
    }

    /** Determine if the given path is local and can be used in the Python snippet */
    private static boolean isLocalPath(final Path path) {
        return path.toUri().getScheme().equals("file");
    }
}
