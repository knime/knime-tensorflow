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

import java.net.URL;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.core.DLTensorSpec;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SavedModel;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

/**
 * Wrapper for TensorFlow {@link SavedModel}. Can read them from a file or
 * directory and extract important information.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowSavedModel {

	private final SavedModel m_savedModel;

	/**
	 * Wraps a SavedModel at the given location to an
	 * {@link DLTensorFlowSavedModel}.
	 *
	 * @param source
	 *            URL to the SavedModel directory or zip file. The directory
	 *            must be a valid SavedModel as defined <a href=
	 *            "https://www.tensorflow.org/programmers_guide/saved_model#structure_of_a_savedmodel_directory">here</a>.
	 *            A zip file must contain such a SavedModel directory.
	 * @throws DLInvalidSourceException
	 *             if the SavedModel coudln't be read
	 */
	public DLTensorFlowSavedModel(URL source) throws DLInvalidSourceException {
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
	 * Extracts the signature names from the SavedModel.
	 *
	 * @param tags
	 *            the tags to consider
	 * @return a list of signature names
	 */
	public Collection<String> getSignatureDefs(Collection<String> tags) {
		return m_savedModel.getMetaGraphsList().stream()
				.filter(m -> m.getMetaInfoDef().getTagsList().stream().anyMatch(tags::contains))
				.flatMap(m -> m.getSignatureDefMap().keySet().stream()).collect(Collectors.toSet());
	}

	/**
	 * Creates the specs for a DLNetwork with this SavedModel.
	 *
	 * @param tags
	 *            the tags to consider.
	 * @param signatures
	 *            the signatures to consider.
	 * @return the specs.
	 */
	public DLTensorFlowSavedModelNetworkSpec createSpecs(final String[] tags, final String[] signatures) {
		List<String> tagsList = Arrays.asList(tags);
		List<String> signaturesList = Arrays.asList(signatures);

		List<MetaGraphDef> metaGraphDefs = m_savedModel.getMetaGraphsList();

		// Get the signature definitions of the selected tags and signatures
		Set<SignatureDef> signatureDefs = metaGraphDefs.stream()
				.filter(m -> m.getMetaInfoDef().getTagsList().stream().anyMatch(tagsList::contains))
				.flatMap(m -> m.getSignatureDefMap().entrySet().stream()
						.filter(e -> signaturesList.contains(e.getKey())).map(e -> e.getValue()))
				.collect(Collectors.toSet());

		// Get the inputs and outputs from the signature definitions
		List<Entry<String, TensorInfo>> inputs = signatureDefs.stream()
				.flatMap(s -> s.getInputsMap().entrySet().stream()).collect(Collectors.toList());
		List<Entry<String, TensorInfo>> outputs = signatureDefs.stream()
				.flatMap(s -> s.getOutputsMap().entrySet().stream()).collect(Collectors.toList());

		// Create DLTensorSpec for the inputs and outputs
		DLTensorSpec[] inputSpecs = new DLTensorSpec[0];
		DLTensorSpec[] hiddenSpecs = new DLTensorSpec[0];
		DLTensorSpec[] outputSpecs = new DLTensorSpec[0];
		// TODO this is affected by the refactoring

		// Create the NetworkSpec
		return new DLTensorFlowSavedModelNetworkSpec(tags, inputSpecs, hiddenSpecs, outputSpecs);
	}
}
