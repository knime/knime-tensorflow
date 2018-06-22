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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Enumeration;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;

import org.apache.commons.io.FileUtils;
import org.knime.core.data.filestore.FileStore;
import org.knime.core.util.FileUtil;
import org.knime.dl.core.DLInvalidSourceException;
import org.tensorflow.framework.SavedModel;

/**
 * Utility class for handling TensorFlow SavedModels.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFSavedModelUtil {

	private static final String SAVED_MODEL_REGEX = "^.*saved_model.pb$" + "|^.*variables(/.*|\\.*)?$"
			+ "|^.*assets(/.*|\\.*)?$";

	/** Map saving which files already have been extracted to a folder. */
	private static final Map<URL, File> CACHED_MODELS = new ConcurrentHashMap<>();

	private TFSavedModelUtil() {
		// Utility class
	}

	private static enum SavedModelType {
		LOCAL_DIR, LOCAL_ZIP, REMOTE_ZIP;
	}

	/**
	 * Deletes the temporary directory for the given source if it points to a local zip file which has been extracted
	 * before.
	 *
	 * @param source a URL pointing to the model source
	 */
	public static void deleteTempIfLocal(final URL source) {
		if (!getSavedModelType(source).equals(SavedModelType.LOCAL_ZIP)) {
			// If it's not a local zip: Do nothing
			return;
		}
		final File tmp = CACHED_MODELS.remove(source);
		if (tmp != null) {
			FileUtil.deleteRecursively(tmp);
		}
	}

	/**
	 * Reads the {@link SavedModel} inside the given zip file or directory. The directory must be a valid SavedModel as
	 * defined
	 * <a href= "https://www.tensorflow.org/programmers_guide/saved_model#structure_of_a_savedmodel_directory">here</a>.
	 * A zip file must contain such a SavedModel directory. For remote models the file gets downloaded and extracted
	 * only if this is the first time accessing this model.
	 *
	 * @param source URL to the SavedModel directory or zip file
	 * @return the SavedModel
	 * @throws DLInvalidSourceException if the SavedModel coudln't be read
	 */
	public static SavedModel readSavedModelProtoBuf(final URL source) throws DLInvalidSourceException {
		try {
			switch (getSavedModelType(source)) {
			case LOCAL_DIR:
				return readSavedModelFromDir(getSavedModelInDir(source));

			case REMOTE_ZIP:
				// Let's get a directory with the SavedModel and read it from there
				final File tmp = getSavedModelInDir(source);
				try {
					return readSavedModelFromDir(tmp);
				} catch (final DLInvalidSourceException e) {
					// Delete the temp directory
					CACHED_MODELS.remove(source);
					FileUtil.deleteRecursively(tmp);
					throw e;
				}

			case LOCAL_ZIP:
				// We can read it more efficiently than remote files using ZipFile
				return readSavedModelFromLocalZip(FileUtil.getFileFromURL(source));

			default:
				// We know that we handled all cases
				return null;
			}
		} catch (final ZipException e) {
			throw new DLInvalidSourceException("Could not read the SavedModel ZIP file.", e);
		} catch (final IOException e) {
			throw new DLInvalidSourceException("Could not read the SavedModel.", e);
		}
	}

	/**
	 * Copies the SavedModel at the given source to the given FileStore.
	 *
	 * @param source a URL pointing to the SavedModel
	 * @param destination the FileStore
	 * @throws IOException if copying the model fails
	 */
	public static void copySavedModelToFileStore(final URL source, final FileStore destination) throws IOException {
		final File destinationFile = destination.getFile();

		switch (getSavedModelType(source)) {
		case LOCAL_DIR:
			copyDirToFile(FileUtil.getFileFromURL(source), destinationFile);
			break;

		case LOCAL_ZIP:
			extractZipToFile(source, destinationFile);
			break;

		case REMOTE_ZIP:
			extractZipToFile(source, destinationFile);
			break;
		}
	}

	/**
	 * Gives a directory where the SavedModel at the given source can be read from. If the source points to a local
	 * directory this function will only return this directory. If it points to a zip file (local or remote) it will
	 * extract the file to a directory. If the ZIP file has been extracted before the same directory is returned.
	 *
	 * @param source a URL pointing to the SavedModel
	 * @return a file pointing to a directory containing the SavedModel
	 * @throws IOException if reading the SavedModel failed
	 */
	public static File getSavedModelInDir(final URL source) throws IOException {
		switch (getSavedModelType(source)) {
		case LOCAL_DIR:
			return FileUtil.getFileFromURL(source);

		case LOCAL_ZIP:
		case REMOTE_ZIP:
			// Check if the zip file already has been extracted
			if (CACHED_MODELS.containsKey(source)) {
				return CACHED_MODELS.get(source);
			}
			// Extract the zip file and remember
			// TODO the KNIME API description is wrong here:
			// final File extracted = FileUtil.createTempDir("SavedModel");
			final File extracted = FileUtil.createTempDir("SavedModel", FileUtils.getTempDirectory());
			try {
				extractZipToFile(source, extracted);
			} catch (final IOException e) {
				// Delete the temp directory
				CACHED_MODELS.remove(source);
				FileUtil.deleteRecursively(extracted);
				throw e;
			}
			CACHED_MODELS.put(source, extracted);
			return extracted;

		default:
			// We know that we handled all cases
			return null;
		}
	}

	/**
	 * Checks if the given URL points to a local directory.
	 *
	 * @param source URL to check
	 * @return true if the URL points to a local directory
	 */
	private static SavedModelType getSavedModelType(final URL source) {
		if (source.getProtocol().equalsIgnoreCase("file") || source.getProtocol().equalsIgnoreCase("knime")) {
			if (FileUtil.getFileFromURL(source).isDirectory()) {
				return SavedModelType.LOCAL_DIR;
			} else {
				return SavedModelType.LOCAL_ZIP;
			}
		} else {
			return SavedModelType.REMOTE_ZIP;
		}
	}

	/**
	 * Extracts the zip file at the given source URL to the destination which should be an empty directory.
	 *
	 * @param source the source URL (Every URL where KNIME can open an InputStream on)
	 * @param destination the destination file
	 */
	private static void extractZipToFile(final URL source, final File destination) throws IOException {
		createDirs(destination);

		// The prefix from the root of the zip file to the SavedModel
		String prefix = "";

		// Extract the zip file
		try (final InputStream fileStream = FileUtil.openStreamWithTimeout(source)) {
			try (final ZipInputStream zipStream = new ZipInputStream(fileStream)) {
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
					final File destFile = new File(destination, name);
					if (entry.isDirectory()) {
						createDirs(destFile);
					} else {
						createDirs(destFile.getParentFile());
						FileUtils.copyToFile(zipStream, destFile);
					}
				}
			}
		}

		// Move the SavedModel to the root of the FileStore
		final File savedModelDir = new File(destination, prefix);
		if (!savedModelDir.equals(destination)) {
			for (final File f : savedModelDir.listFiles()) {
				FileUtils.moveToDirectory(f, destination, false);
			}

			// Delete everything else
			for (final File f : destination.listFiles()) {
				if (!f.getName().matches(SAVED_MODEL_REGEX)) {
					FileUtil.deleteRecursively(f);
				}
			}
		}
	}

	/**
	 * Creates a directory at the given File location if it doesn't exist already.
	 *
	 * @param dir the directory which should be created
	 * @throws IOException if creating the directory failed
	 */
	private static void createDirs(final File dir) throws IOException {
		if (!dir.isDirectory() && !dir.mkdirs()) {
			throw new IOException(
					"Cannot create destination directory \"" + dir.getAbsolutePath() + "\" for the SavedModel.");
		}
	}

	/**
	 * Copies the relevant files of a SavedModel to another directory. Assumes that the source is a directory and
	 * exists.
	 *
	 * @param source the source directory
	 * @param destination the destination directory
	 * @throws IOException if copying failed
	 */
	private static void copyDirToFile(final File source, final File destination) throws IOException {
		if (!destination.equals(source)) {
			// Create the target directory if it doesn't exist yet
			createDirs(destination);
			final String[] sourceList = source.list();
			if (sourceList == null) {
				throw new IOException(
						"Can't copy SavedModel directory \"" + source.getAbsolutePath() + "\", no read permissions.");
			}
			for (final String child : sourceList) {
				// Only copy the child if it is part of the SavedModel
				// definition
				if (child.matches(SAVED_MODEL_REGEX)) {
					FileUtil.copyDir(new File(source, child), new File(destination, child));
				}
			}
		}
	}

	private static SavedModel readSavedModelFromDir(final File file) throws DLInvalidSourceException {
		try {
			final File[] savedModelPb = file.listFiles((d, n) -> n.equals("saved_model.pb"));
			if (savedModelPb.length == 0) {
				if (file.listFiles((d, n) -> n.equals("saved_model.pbtxt")).length > 0) {
					throw new DLInvalidSourceException(
							"The SavedModel is stored in the non supported pbtxt format. Please save your model with a saved_model.pb");
				} else {
					throw new DLInvalidSourceException("The directory doesn't contain a saved_model.pb");
				}
			}
			try (final FileInputStream inStream = new FileInputStream(savedModelPb[0])) {
				return SavedModel.parseFrom(inStream);
			}
		} catch (FileNotFoundException e) {
			throw new DLInvalidSourceException("The directory doesn't contain a saved_model.pb");
		} catch (IOException e) {
			throw new DLInvalidSourceException("The SavedModel could not be parsed.", e);
		}
	}

	/**
	 * Reads a SavedModel from a local file.
	 *
	 * @param file the local zip file
	 * @return the SavedModel
	 */
	private static SavedModel readSavedModelFromLocalZip(final File file)
			throws ZipException, IOException, DLInvalidSourceException {
		try (ZipFile savedModelZip = new ZipFile(file)) {
			final Enumeration<? extends ZipEntry> entries = savedModelZip.entries();
			ZipEntry entry = null;
			boolean hasPBTXT = false;
			while (entries.hasMoreElements()) {
				final ZipEntry zipEntry = entries.nextElement();
				if (zipEntry.getName().endsWith("saved_model.pb")) {
					entry = zipEntry;
				} else if (zipEntry.getName().endsWith("saved_model.pbtxt")) {
					// Remember that there is a pbtxt to warn the user (but maybe we will still find an pb)
					hasPBTXT = true;
				}
			}
			if (entry == null) {
				if (hasPBTXT) {
					throw new DLInvalidSourceException(
							"The SavedModel is stored in the non supported pbtxt format. Please save your model with a saved_model.pb");
				} else {
					throw new DLInvalidSourceException("The zip file doesn't contain a saved_model.pb");
				}
			}
			return SavedModel.parseFrom(savedModelZip.getInputStream(entry));
		}
	}
}
