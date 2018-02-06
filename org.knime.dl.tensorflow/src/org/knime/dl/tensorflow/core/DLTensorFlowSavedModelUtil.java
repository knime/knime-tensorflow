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
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

import org.knime.core.util.FileUtil;
import org.knime.dl.core.DLInvalidSourceException;
import org.tensorflow.framework.SavedModel;

/**
 * Utility class for handling TensorFlow SavedModels.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowSavedModelUtil {

	private DLTensorFlowSavedModelUtil() {
		// Utility class
	}

	/**
	 * Reads the {@link SavedModel} inside the given zip file or directory. The directory must be a valid SavedModel as
	 * defined
	 * <a href= "https://www.tensorflow.org/programmers_guide/saved_model#structure_of_a_savedmodel_directory">here</a>.
	 * A zip file must contain such a SavedModel directory.
	 *
	 * @param source URL to the SavedModel directory or zip file
	 * @return the SavedModel
	 * @throws DLInvalidSourceException if the SavedModel coudln't be read
	 */
	public static SavedModel readSavedModelProtoBuf(final URL source) throws DLInvalidSourceException {
		final File file = FileUtil.getFileFromURL(source);
		if (file.isDirectory()) {
			try {
				final File[] savedModelPb = file
						.listFiles((d, n) -> n.equals("saved_model.pb") || n.equals("saved_model.pbtxt"));
				if (savedModelPb.length == 0) {
					throw new DLInvalidSourceException("The directory doesn't contain a saved_model.pb");
				}
				return SavedModel.parseFrom(new FileInputStream(savedModelPb[0]));
			} catch (final FileNotFoundException e) {
				throw new DLInvalidSourceException("The directory doesn't contain a saved_model.pb", e);
			} catch (final IOException e) {
				throw new DLInvalidSourceException("The SavedModel could not be parsed.", e);
			}
		} else {
			try (ZipFile savedModelZip = new ZipFile(file)) {
				final Enumeration<? extends ZipEntry> entries = savedModelZip.entries();
				ZipEntry entry = null;
				while (entries.hasMoreElements()) {
					final ZipEntry zipEntry = entries.nextElement();
					if (zipEntry.getName().endsWith("saved_model.pb")
							|| zipEntry.getName().endsWith("saved_model.pdtxt")) {
						entry = zipEntry;
					}
				}
				if (entry == null) {
					throw new DLInvalidSourceException("The zip file doesn't contain a saved_model.pb");
				}
				final InputStream inputStream = savedModelZip.getInputStream(entry);
				return SavedModel.parseFrom(inputStream);
			} catch (final ZipException e) {
				throw new DLInvalidSourceException("This SavedModel zip file isn't a valid zip file.", e);
			} catch (final IOException e) {
				throw new DLInvalidSourceException("The SavedModel file could not be read.", e);
			}
		}
	}
}
