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
package org.knime.dl.tensorflow.savedmodel.core;

import java.io.IOException;
import java.net.URL;
import java.util.Collections;
import java.util.List;

import org.knime.base.filehandling.remote.files.RemoteFileHandlerRegistry;
import org.knime.core.data.filestore.FileStore;
import org.knime.dl.core.DLInvalidDestinationException;
import org.knime.dl.core.DLInvalidEnvironmentException;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.python.core.DLPythonAbstractCommands;
import org.knime.dl.python.core.DLPythonAbstractNetworkLoader;
import org.knime.dl.python.core.DLPythonContext;
import org.knime.dl.python.core.DLPythonNetwork;
import org.knime.dl.python.core.DLPythonNetworkHandle;
import org.knime.dl.python.core.DLPythonNetworkPortObject;

/**
 * TODO Abstract class for all tensorflow python networks?
 * 
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFPythonNetworkLoader extends DLPythonAbstractNetworkLoader<TFSavedModelNetwork> {

	private static final String URL_EXTENSION = "";

	@Override
	public Class<TFSavedModelNetwork> getNetworkType() {
		return TFSavedModelNetwork.class;
	}

	@Override
	public String getPythonModuleName() {
		return "TFNetworkType";
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
	public void validateSource(final URL source) throws DLInvalidSourceException {
		// TODO validate the source: Maybe use the java SavedModel helper classes
	}

	@Override
	public void validateDestination(final URL destination) throws DLInvalidDestinationException {
		// TODO copied from DLKerasAbstractNetworkLoader. Does this also work for this case?
		try {
			RemoteFileHandlerRegistry.getRemoteFileHandler(destination.getProtocol())
					.createRemoteFile(destination.toURI(), null, null);
		} catch (final Exception e) {
			throw new DLInvalidDestinationException(
					"An error occurred while resolving the Keras network file location.\nCause: " + e.getMessage(), e);
		}
	}

	@Override
	public DLPythonNetworkHandle load(final URL source, final DLPythonContext context, final boolean loadTrainingConfig)
			throws DLInvalidSourceException, DLInvalidEnvironmentException, IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public TFSavedModelNetwork fetch(final DLPythonNetworkHandle handle, final URL source,
			final DLPythonContext context)
			throws IllegalArgumentException, DLInvalidSourceException, DLInvalidEnvironmentException, IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DLPythonNetworkPortObject<? extends DLPythonNetwork> createPortObject(final TFSavedModelNetwork network,
			final FileStore fileStore) throws IOException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected DLPythonAbstractCommands createCommands(final DLPythonContext context)
			throws DLInvalidEnvironmentException {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	protected DLPythonInstallationTester getInstallationTester() {
		// TODO Auto-generated method stub
		return null;
	}
}
