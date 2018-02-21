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

import java.net.URL;
import java.util.Collection;
import java.util.stream.Collectors;

import org.knime.dl.core.DLInvalidSourceException;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SavedModel;

/**
 * Wrapper for TensorFlow {@link SavedModel}. Can read them from a file or directory and extract important information.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowSavedModel {

	private final SavedModel m_savedModel;

	/**
	 * Wraps a SavedModel at the given location to an {@link DLTensorFlowSavedModel}.
	 *
	 * @param source URL to the SavedModel directory or zip file. The directory must be a valid SavedModel as defined
	 *            <a href=
	 *            "https://www.tensorflow.org/programmers_guide/saved_model#structure_of_a_savedmodel_directory">here</a>.
	 *            A zip file must contain such a SavedModel directory.
	 * @throws DLInvalidSourceException if the SavedModel coudln't be read
	 */
	public DLTensorFlowSavedModel(final URL source) throws DLInvalidSourceException {
		m_savedModel = DLTensorFlowSavedModelUtil.readSavedModelProtoBuf(source);
	}

	/**
	 * Extracts the tags from the SavedModel.
	 *
	 * @return a list of the tags
	 */
	public Collection<String> getContainedTags() {
		return m_savedModel.getMetaGraphsList().stream().flatMap(m -> m.getMetaInfoDef().getTagsList().stream())
				.collect(Collectors.toSet());
	}

	/**
	 * Get a {@link DLTensorFlowMetaGraphDefs} with the {@link MetaGraphDef}s of this SavedModel considering the given
	 * tags.
	 *
	 * @param tags the tags to consider
	 * @return a {@link DLTensorFlowMetaGraphDefs} with the {@link MetaGraphDef}s of this SavedModel
	 */
	public DLTensorFlowMetaGraphDefs getMetaGraphDefs(final Collection<String> tags) {
		return new DLTensorFlowMetaGraphDefs(m_savedModel, tags);
	}
}
