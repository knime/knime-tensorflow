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
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.knime.dl.core.DLAbstractNetworkSpec;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.core.DLNetworkSpec;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.tensorflow.core.training.DLTensorFlowTrainingConfig;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SavedModel;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

/**
 * The spec of a {@link DLTensorFlowSavedModelNetwork}.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowSavedModelNetworkSpec extends DLAbstractNetworkSpec<DLTensorFlowTrainingConfig>
		implements DLTensorFlowNetworkSpec {

	private static final long serialVersionUID = 1L;
	
	private final String[] m_tags;

	protected DLTensorFlowSavedModelNetworkSpec(final String[] tags, final DLTensorSpec[] inputSpecs, final DLTensorSpec[] hiddenOutputSpecs,
			final DLTensorSpec[] outputSpecs) {
		super(inputSpecs, hiddenOutputSpecs, outputSpecs);
		m_tags = tags;
	}

	protected DLTensorFlowSavedModelNetworkSpec(final String[] tags, final DLTensorSpec[] inputSpecs, final DLTensorSpec[] hiddenOutputSpecs,
			final DLTensorSpec[] outputSpecs, DLTensorFlowTrainingConfig trainingConfig) {
		super(inputSpecs, hiddenOutputSpecs, outputSpecs, trainingConfig);
		m_tags = tags;
	}

	@Override
	protected void hashCodeInternal(HashCodeBuilder b) {
		b.append(m_tags);
	}

	@Override
	protected boolean equalsInternal(DLNetworkSpec other) {
		DLTensorFlowSavedModelNetworkSpec o = (DLTensorFlowSavedModelNetworkSpec) other;
		if (m_tags.length != o.m_tags.length) {
			return false;
		}
		// The order of the tags doesn't matter
		return Arrays.asList(m_tags).containsAll(Arrays.asList(o.m_tags));
	}

	@Override
	public DLTensorFlowNetwork create(URL source) throws DLInvalidSourceException {
		return new DLTensorFlowSavedModelNetwork(this, source);
	}
	
	public String[] getTags() {
		return m_tags;
	}

	public static DLTensorFlowSavedModelNetworkSpec createSpecs(final SavedModel sm, final String[] tags,
			final String[] signatures) {
		List<String> tagsList = Arrays.asList(tags);
		List<String> signaturesList = Arrays.asList(signatures);

		List<MetaGraphDef> metaGraphDefs = DLTensorFlowSavedModelUtil.getMetaGraphDefs(sm);

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
