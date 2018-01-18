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
import java.io.InputStream;
import java.net.URL;
import java.util.Collections;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import org.apache.commons.io.FileUtils;
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

	private static final String SAVED_MODEL_REGEX = "^.*saved_model.pb(txt)?$" + "|^.*variables(/.*|\\.*)?$"
			+ "|^.*assets(/.*|\\.*)?$";

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
			createDirs(destination);
			final String[] sourceList = source.list();
			if (sourceList == null) {
				throw new IOException(
						"Can't copy SavedModel directory \"" + source.getAbsolutePath() + "\", no read permissions.");
			}
			for (String child : sourceList) {
				// Only copy the child if it is part of the SavedModel
				// definition
				if (child.matches(SAVED_MODEL_REGEX)) {
					FileUtil.copyDir(new File(source, child), new File(destination, child));
				}
			}
		}
	}

	private void extractZipToFileStore(File source, File destination) throws IOException {
		createDirs(destination);
		try (final ZipFile zip = new ZipFile(source)) {
			final String prefix = getZipPrefix(zip);

			Enumeration<? extends ZipEntry> entries = zip.entries();
			while (entries.hasMoreElements()) {
				final ZipEntry e = entries.nextElement();

				// Check if the entry is a relevant file in the SavedModel we
				// extract
				final String name = e.getName();
				if (!name.matches(SAVED_MODEL_REGEX) || !name.startsWith(prefix)) {
					continue;
				}

				// Extract this entry
				final File destFile = new File(destination, name.substring(prefix.length()));
				if (e.isDirectory()) {
					createDirs(destFile);
				} else {
					createDirs(destFile.getParentFile());
					InputStream inputStream = zip.getInputStream(e);
					FileUtils.copyInputStreamToFile(inputStream, destFile);
				}
			}
		}
	}

	private void createDirs(final File dir) throws IOException {
		if (!dir.isDirectory() && !dir.mkdirs()) {
			throw new IOException(
					"Cannot create destination directory \"" + dir.getAbsolutePath() + "\" for the SavedModel.");
		}
	}

	private String getZipPrefix(final ZipFile zip) throws IOException {
		Enumeration<? extends ZipEntry> entries = zip.entries();
		ZipEntry savedModelZip = Collections.list(entries).stream()
				.filter(e -> e.getName().matches("^.*saved_model.pb(txt)?$")).findFirst()
				.orElseThrow(() -> new IOException("The zip file is not a valid SavedModel"));
		int prefixLength = savedModelZip.getName().lastIndexOf("saved_model.pb");
		return savedModelZip.getName().substring(0, prefixLength);
	}
}
