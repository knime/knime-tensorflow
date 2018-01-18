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
package org.knime.dl.tensorflow.core;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Collection;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.knime.core.data.filestore.FileStore;
import org.knime.core.util.FileUtil;
import org.knime.dl.python.core.DLPythonAbstractNetwork;

/**
 * Deep learning network for TensorFlow using the SavedModel format.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowSavedModelNetwork extends DLPythonAbstractNetwork<DLTensorFlowSavedModelNetworkSpec>
		implements DLTensorFlowNetwork {

	private static final Collection<String> SAVED_MODEL_FILES = Stream
			.of("saved_model.pb", "saved_model.pbtxt", "variables", "assets").collect(Collectors.toSet());

	protected DLTensorFlowSavedModelNetwork(DLTensorFlowSavedModelNetworkSpec spec, URL source) {
		super(spec, source);
	}

	@Override
	public void copyRelevantToFileStore(final FileStore destination) throws IOException {
		final File sourceFile = FileUtil.getFileFromURL(getSource());
		final File destinationFile = destination.getFile();
		if (sourceFile.isDirectory()) {
			copyDirToFileStore(sourceFile, destinationFile);
		} else {
			extractZipToFileStore(sourceFile, destinationFile);
		}
	}

	/**
	 * Copies the relevant files of a SavedModel to another directory. Asumes
	 * that the source is a directory and exists.
	 *
	 * @param source
	 *            the source directory
	 * @param destination
	 *            the destination directory
	 * @throws IOException
	 *             if copying failed
	 */
	private void copyDirToFileStore(final File source, final File destination) throws IOException {
		if (!destination.toURI().toURL().equals(getSource())) {
			// Create the target directory if it doesn't exist yet
			if (!destination.isDirectory() && !destination.mkdirs()) {
				throw new IOException("Cannot create destination directory \"" + destination.getAbsolutePath()
						+ "\" for the SavedModel.");
			}
			final String[] sourceList = source.list();
			if (sourceList == null) {
				throw new IOException(
						"Can't copy SavedModel directory \"" + source.getAbsolutePath() + "\", no read permissions.");
			}
			for (String child : sourceList) {
				// Only copy the child if it is part of the SavedModel definition
				if (SAVED_MODEL_FILES.contains(child)) {
					FileUtil.copyDir(new File(source, child), new File(destination, child));
				}
			}
		}
	}

	private void extractZipToFileStore(File source, File destination) throws IOException {
		// TODO limit on relevant files. Use SavedModel definition
		// TODO manage zips with different directory structure: remove leading
		// paths
		destination.mkdirs();
		FileUtil.unzip(source, destination);
	}
}
