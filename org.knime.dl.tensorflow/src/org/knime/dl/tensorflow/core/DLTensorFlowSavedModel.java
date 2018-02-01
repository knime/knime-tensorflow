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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.OptionalLong;
import java.util.stream.Collectors;

import org.knime.dl.core.DLDefaultFixedTensorShape;
import org.knime.dl.core.DLDefaultPartialTensorShape;
import org.knime.dl.core.DLDefaultTensorId;
import org.knime.dl.core.DLDefaultTensorSpec;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorShape;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.DLUnknownTensorShape;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.NodeDef;
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
	 *            URL to the SavedModel directory or zip file. The directory must be
	 *            a valid SavedModel as defined <a href=
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
	public Collection<String> getSignatureDefsStrings(Collection<String> tags) {
		return getSignatureDefsWithoutSerializedTensors(tags).stream().map(e -> e.getKey()).collect(Collectors.toSet());
	}

	/**
	 * Extracts the tensor specifications of all tensors which may can be used as
	 * inputs to the graph.
	 *
	 * @param tags
	 *            the tags to consider
	 * @return a collection of tensor specifications
	 */
	public Collection<DLTensorSpec> getPossibleInputTensors(final Collection<String> tags) {
		return getFilteredMetagraphDefs(tags).stream().flatMap(m -> m.getGraphDef().getNodeList().stream()
				.filter(n -> nodeDefCanBeInput(n)).map(n -> getTensorSpecFromNodeDef(n))).collect(Collectors.toSet());
	}

	/**
	 * Extracts the tensor specifications of all tensors which may can be used as
	 * outputs from the graph.
	 *
	 * @param tags
	 *            the tags to consider
	 * @return a collection of tensor specifications
	 */
	public Collection<DLTensorSpec> getPossibleOutputTensors(final Collection<String> tags) {
		return getFilteredMetagraphDefs(tags).stream().flatMap(m -> m.getGraphDef().getNodeList().stream()
				.filter(n -> nodeDefCanBeOutput(n)).map(n -> getTensorSpecFromNodeDef(n))).collect(Collectors.toSet());
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
	public DLTensorFlowSavedModelNetworkSpec createSpecs(final String[] tags, final String signature) {
		List<String> tagsList = Arrays.asList(tags);

		// Get the signature definitions of the selected tags and signatures
		SignatureDef signatureDef = getFilteredMetagraphDefs(tagsList).stream().flatMap(m -> m.getSignatureDefMap()
				.entrySet().stream().filter(e -> signature.equals(e.getKey())).map(e -> e.getValue())).findFirst()
				.get(); // TODO throw exception if not present

		// Get the inputs and outputs from the signature definitions
		Map<String, TensorInfo> inputs = signatureDef.getInputsMap();
		Map<String, TensorInfo> outputs = signatureDef.getOutputsMap();

		// Create DLTensorSpec for the inputs and outputs
		DLTensorSpec[] inputSpecs = new DLTensorSpec[0];
		DLTensorSpec[] hiddenSpecs = new DLTensorSpec[0];
		DLTensorSpec[] outputSpecs = new DLTensorSpec[0];
		// TODO this is affected by the refactoring

		// Create the NetworkSpec
		return new DLTensorFlowSavedModelNetworkSpec(tags, inputSpecs, hiddenSpecs, outputSpecs);
	}

	private Collection<MetaGraphDef> getFilteredMetagraphDefs(final Collection<String> tags) {
		return m_savedModel.getMetaGraphsList().stream()
				.filter(m -> m.getMetaInfoDef().getTagsList().stream().anyMatch(tags::contains))
				.collect(Collectors.toList());
	}

	private Collection<Entry<String, SignatureDef>> getSignatureDefsWithoutSerializedTensors(
			final Collection<String> tags) {
		return getSignatureDefs(tags).stream().filter(e -> !containsSerializedTensors(e.getValue()))
				.collect(Collectors.toSet());
	}

	private Collection<Entry<String, SignatureDef>> getSignatureDefs(final Collection<String> tags) {
		return getFilteredMetagraphDefs(tags).stream().flatMap(m -> m.getSignatureDefMap().entrySet().stream())
				.collect(Collectors.toSet());
	}

	private boolean containsSerializedTensors(final SignatureDef signatureDef) {
		return signatureDef.getInputsMap().entrySet().stream().anyMatch(t -> isSerializedTensor(t.getValue()))
				|| signatureDef.getOutputsMap().entrySet().stream().anyMatch(t -> isSerializedTensor(t.getValue()));
	}

	private boolean isSerializedTensor(final TensorInfo t) {
		// TODO not really every String tensor is a serialized tensor but as we do not
		// support String inputs yet it is fine
		return t.getDtype().equals(DataType.DT_STRING);
	}

	private DLTensorSpec getTensorSpecFromNodeDef(final NodeDef n) {
		// Get the type
		final Class<?> type = getDataTypeOfNodeDef(n);
		// Get the name
		final String name = n.getName();
		// The names should work as identifiers in a TensorFlow graph
		final DLTensorId id = new DLDefaultTensorId(name);

		// Get the shape and batch size
		try {
			final AttrValue shapeAttr = n.getAttrOrThrow("shape");
			final List<Long> shapeList = new ArrayList<>(
					shapeAttr.getShape().getDimList().stream().map(d -> d.getSize()).collect(Collectors.toList()));
			if (shapeList.size() > 0) {
				// Get the batch size
				final long batchSize = shapeList.get(0);
				shapeList.remove(0);

				// Create the shape
				DLTensorShape shape;
				if (shapeList.isEmpty()) {
					// No shape other than the batch size
					shape = DLUnknownTensorShape.INSTANCE;
				} else if (shapeList.contains(-1L)) {
					// At least one dimension is unknown
					OptionalLong[] dims = shapeList.stream().map(l -> l > 0 ? OptionalLong.of(l) : OptionalLong.empty())
							.toArray(OptionalLong[]::new);
					shape = new DLDefaultPartialTensorShape(dims);
				} else {
					// The shape is fixed
					long[] dims = shapeList.stream().mapToLong(Long::longValue).toArray();
					shape = new DLDefaultFixedTensorShape(dims);
				}
				if (batchSize > 0) {
					return new DLDefaultTensorSpec(id, name, batchSize, shape, type);
				} else {
					return new DLDefaultTensorSpec(id, name, shape, type);
				}
			}
		} catch (IllegalArgumentException e) {
			// nothing to do
		}

		// Getting the shape and batch size wasn't successful
		return new DLDefaultTensorSpec(id, name, type);
	}

	private boolean nodeDefCanBeInput(final NodeDef n) {
		return getDataTypeOfNodeDef(n) != null; // TODO not everything can be an input, right?
	}

	private boolean nodeDefCanBeOutput(final NodeDef n) {
		return getDataTypeOfNodeDef(n) != null; // TODO not everything can be an output, right?
	}

	private Class<?> getDataTypeOfNodeDef(final NodeDef n) {
		try {
			final AttrValue typeAttr = n.getAttrOrThrow("dtype");
			return Float.class; // TODO implement
		} catch (IllegalArgumentException e) {
			return null; // Couldn't find a data type
		}
	}
}
