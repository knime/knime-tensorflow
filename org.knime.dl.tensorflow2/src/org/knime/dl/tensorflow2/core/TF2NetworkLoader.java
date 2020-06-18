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

import static com.google.common.base.Preconditions.checkNotNull;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

import org.knime.base.filehandling.remote.files.RemoteFileHandlerRegistry;
import org.knime.core.data.filestore.FileStore;
import org.knime.dl.core.DLCancelable;
import org.knime.dl.core.DLCanceledExecutionException;
import org.knime.dl.core.DLInvalidDestinationException;
import org.knime.dl.core.DLInvalidEnvironmentException;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.core.DLNetworkLocation;
import org.knime.dl.python.core.DLPythonAbstractNetworkLoader;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonNetwork;
import org.knime.dl.python.core.DLPythonNetworkHandle;
import org.knime.dl.python.core.DLPythonNetworkPortObject;
import org.knime.dl.tensorflow2.base.portobjects.TF2NetworkPortObject;

/**
 * The loader for a {@link TF2Network}.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TF2NetworkLoader extends DLPythonAbstractNetworkLoader<TF2Network> {

    private static final String URL_EXTENSION = "";

    private static final DLPythonInstallationTester INSTALLATION_TESTER =
        new DLPythonInstallationTester(() -> new TF2PythonContext());

    @Override
    public Class<TF2Network> getNetworkType() {
        return TF2Network.class;
    }

    @Override
    public String getPythonModuleName() {
        return "TF2NetworkType";
    }

    @Override
    public List<String> getLoadModelURLExtensions() {
        return Collections.singletonList(URL_EXTENSION);
    }

    @Override
    public String getSaveModelURLExtension() {
        return URL_EXTENSION;
    }

    @Override
    public URL validateSource(final URI source) throws DLInvalidSourceException {
        final URL sourceURL;
        try {
            sourceURL = source.toURL();
        } catch (final Exception e) {
            throw new DLInvalidSourceException("TensorFlow network source (" + source + ") is not a valid URL.");
        }
        checkIfLocal(source);
        try {
            if (!Files.list(Paths.get(source))
                .anyMatch(p -> p.getFileName().toString().matches("^.*saved_model.pb?$"))) {
                throw new DLInvalidSourceException("The source does not contain a saved_model.pb file.");
            }
        } catch (final IOException ex) {
            throw new DLInvalidSourceException("The source could not be read.", ex);
        }

        return sourceURL;
    }

    @Override
    public URL validateDestination(final URI destination) throws DLInvalidDestinationException {
        final URL destinationURL;
        try {
            destinationURL = destination.toURL();
        } catch (final Exception e) {
            throw new DLInvalidDestinationException(
                "TensorFlow network destination (" + destination + ") is not a valid URL.");
        }
        try {
            RemoteFileHandlerRegistry.getRemoteFileHandler(destinationURL.getProtocol()).createRemoteFile(destination,
                null, null);
        } catch (final Exception e) {
            throw new DLInvalidDestinationException(
                "An error occurred while resolving the TensorFlow network file location.\nCause: " + e.getMessage(), e);
        }
        return destinationURL;
    }

    @Override
    @SuppressWarnings("resource") // Commands do not need to be closed because they use the given context
    public DLPythonNetworkHandle load(final URI source, final DLPythonContext context, final boolean loadTrainingConfig,
        final DLCancelable cancelable)
        throws DLInvalidSourceException, DLInvalidEnvironmentException, IOException, DLCanceledExecutionException {
        checkIfLocal(source);
        final File savedModelDir = new File(source);
        final TF2PythonCommands commands = createCommands(checkNotNull(context));
        return commands.loadNetwork(savedModelDir.getAbsolutePath(), loadTrainingConfig, cancelable);
    }

    @Override
    @SuppressWarnings("resource") // Commands do not need to be closed because they use the given context
    public TF2Network fetch(final DLPythonNetworkHandle handle, final DLNetworkLocation source,
        final DLPythonContext context, final DLCancelable cancelable) throws IllegalArgumentException,
        DLInvalidSourceException, DLInvalidEnvironmentException, IOException, DLCanceledExecutionException {
        final TF2PythonCommands commands = createCommands(checkNotNull(context));
        final TF2NetworkSpec spec = commands.extractNetworkSpec(checkNotNull(handle), cancelable);
        return new TF2Network(spec, source);
    }

    @Override
    public DLPythonNetworkPortObject<? extends DLPythonNetwork> createPortObject(final TF2Network network,
        final FileStore fileStore) throws IOException {
        return new TF2NetworkPortObject(network, fileStore);
    }

    @Override
    protected TF2PythonCommands createCommands(final DLPythonContext context) throws DLInvalidEnvironmentException {
        return new TF2PythonCommands(context);
    }

    @Override
    protected DLPythonInstallationTester getInstallationTester() {
        return INSTALLATION_TESTER;
    }

    /** Checks if the URI points to a local file and throws an exception if not */
    private static void checkIfLocal(final URI source) throws DLInvalidSourceException {
        if (!source.getScheme().equals("file")) {
            throw new DLInvalidSourceException("The source \"" + source.toString() + "\" is not a local file.");
        }
    }
}
