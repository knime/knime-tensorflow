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
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.stream.Collectors;

import org.knime.dl.core.DLDefaultDimensionOrder;
import org.knime.dl.core.DLDefaultFixedTensorShape;
import org.knime.dl.core.DLDefaultPartialTensorShape;
import org.knime.dl.core.DLDefaultTensorId;
import org.knime.dl.core.DLDefaultTensorSpec;
import org.knime.dl.core.DLDimensionOrder;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.core.DLTensorId;
import org.knime.dl.core.DLTensorShape;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.DLUnknownTensorShape;
import org.knime.dl.tensorflow.core.DLInvalidTypeException;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.NodeDef;
import org.tensorflow.framework.SavedModel;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;
import org.tensorflow.framework.TensorShapeProto;

/**
 * Wrapper for TensorFlow {@link SavedModel}. Can read them from a file or directory and extract important information.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class DLTensorFlowSavedModel {

	private final SavedModel m_savedModel;

	private Collection<String> m_tags;

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
	 * Sets the tags that should be considered for all other operations.
	 *
	 * @param tags the tags to consider
	 */
	public void setSelectedTags(final Collection<String> tags) {
		// TODO reset stuff that depends on the selected tags
		m_tags = tags;
	}

	/**
	 * Extracts the signature names from the SavedModel.
	 *
	 * @return a list of signature names
	 */
	public Collection<String> getSignatureDefsStrings() {
		return getFilteredSignature().stream().map(e -> e.getKey()).collect(Collectors.toSet());
	}

	/**
	 * Extracts the tensor specifications of all tensors which may can be used as inputs to the graph.
	 *
	 * @return a collection of tensor specifications
	 */
	public Collection<DLTensorSpec> getPossibleInputTensors() {
		return getFilteredMetagraphDefs().stream()
				.flatMap(m -> m.getGraphDef().getNodeList().stream().filter(n -> canBeInput(n))
						.map(n -> createTensorSpec(n, true)))
				.filter(o -> o.isPresent()).map(o -> o.get()).collect(Collectors.toSet());
	}

	/**
	 * Extracts the tensor specifications of all tensors which may can be used as outputs from the graph.
	 *
	 * @return a collection of tensor specifications
	 */
	public Collection<DLTensorSpec> getPossibleOutputTensors() {
		return getFilteredMetagraphDefs().stream()
				.flatMap(m -> m.getGraphDef().getNodeList().stream().filter(n -> canBeOutput(n))
						.map(n -> createTensorSpec(n, false)))
				.filter(o -> o.isPresent()).map(o -> o.get()).collect(Collectors.toSet());
	}

	/**
	 * Creates the specs for a DLNetwork with this SavedModel.
	 *
	 * @param signature the signature to consider which must be available.
	 * @return the specs.
	 */
	public DLTensorFlowSavedModelNetworkSpec createSpecs(final String signature) {
		// Get the signature definitions of the selected tags and signatures
		final SignatureDef signatureDef = getFilteredMetagraphDefs().stream()
				.flatMap(m -> m.getSignatureDefMap().entrySet().stream().filter(e -> signature.equals(e.getKey()))
						.map(e -> e.getValue()))
				.findFirst()
				.orElseThrow(() -> new IllegalArgumentException("The SavedModel doesn't contain the signature."));

		// Get the inputs and outputs from the signature definitions
		final Map<String, TensorInfo> inputs = signatureDef.getInputsMap();
		final Map<String, TensorInfo> outputs = signatureDef.getOutputsMap();

		// Create DLTensorSpec for the inputs and outputs
		final DLTensorSpec[] inputSpecs = inputs.entrySet().stream()
				.map(e -> createTensorSpec(e.getKey(), e.getValue())).toArray(DLTensorSpec[]::new);
		final DLTensorSpec[] hiddenSpecs = new DLTensorSpec[0];
		final DLTensorSpec[] outputSpecs = outputs.entrySet().stream()
				.map(e -> createTensorSpec(e.getKey(), e.getValue())).toArray(DLTensorSpec[]::new);

		// Get the tags as array
		final String[] tags = m_tags.toArray(new String[m_tags.size()]);

		// Create the NetworkSpec
		return new DLTensorFlowSavedModelNetworkSpec(tags, inputSpecs, hiddenSpecs, outputSpecs);
	}

	private Collection<MetaGraphDef> getFilteredMetagraphDefs() {
		return m_savedModel.getMetaGraphsList().stream()
				.filter(m -> m.getMetaInfoDef().getTagsList().stream().anyMatch(m_tags::contains))
				.collect(Collectors.toList());
	}

	private Collection<Entry<String, SignatureDef>> getFilteredSignature() {
		return getSignatureDefs().stream().filter(e -> canBeUsedInKNIME(e.getValue())).collect(Collectors.toSet());
	}

	private Collection<Entry<String, SignatureDef>> getSignatureDefs() {
		return getFilteredMetagraphDefs().stream().flatMap(m -> m.getSignatureDefMap().entrySet().stream())
				.collect(Collectors.toSet());
	}

	private boolean canBeUsedInKNIME(final SignatureDef signatureDef) {
		return signatureDef.getInputsMap().entrySet().stream().anyMatch(t -> canBeInput(t.getValue()))
				&& signatureDef.getOutputsMap().entrySet().stream().anyMatch(t -> canBeOutput(t.getValue()));
	}

	private String opName(final String name) {
		if (name.contains(":")) {
			return name.substring(0, name.lastIndexOf(':'));
		} else {
			return name;
		}
	}

	// ---------------------- Methods on TensorInfo --------------------

	private DLTensorSpec createTensorSpec(final String name, final TensorInfo t) {
		try {
			final Class<?> type = getClassForType(t.getDtype());
			final DLTensorId id = new DLDefaultTensorId(opName(t.getName()));
			final TensorShapeProto shapeProto = t.getTensorShape();
			return createTensorSpec(id, name, shapeProto, type);
		} catch (final DLInvalidTypeException e) {
			throw new IllegalStateException("The chosen tensor has no supported type.", e);
			// This should not happen because we only allow signatures where we know all
			// types
		}
	}

	private boolean canBeInput(final TensorInfo t) {
		return canBeInputOrOutput(t.getDtype());
	}

	private boolean canBeOutput(final TensorInfo t) {
		return canBeInputOrOutput(t.getDtype());
	}

	// ---------------------- Methods on NodeDef --------------------

	private Optional<DLTensorSpec> createTensorSpec(final NodeDef n, final boolean input) {
		// Get the name
		final String name = n.getName();
		// The names should work as identifiers in a TensorFlow graph
		final DLTensorId id = new DLDefaultTensorId(name);

		// Get the type
		final Class<?> type;
		try {
			type = getClassForType(getDataTypeOfNodeDef(n));
		} catch (DLInvalidTypeException e) {
			// This node definition has no type. We cannot create a spec for it
			return Optional.empty();
		}

		// Get the shape
		TensorShapeProto shapeProto = getShapeOfNodeDef(n, input);
		return Optional.of(createTensorSpec(id, name, shapeProto, type));
	}

	private boolean canBeInput(final NodeDef n) {
		final TensorShapeProto shape = getShapeOfNodeDef(n, true);
		return shape != null && !shape.getUnknownRank();
	}

	private boolean canBeOutput(final NodeDef n) {
		final TensorShapeProto shape = getShapeOfNodeDef(n, false);
		return shape != null && !shape.getUnknownRank();
	}

	private DataType getDataTypeOfNodeDef(final NodeDef n) throws DLInvalidTypeException {
		if (n.containsAttr("dtype")) {
			return n.getAttrOrThrow("dtype").getType();
		} else if (n.containsAttr("T")) {
			return n.getAttrOrThrow("T").getType();
		} else {
			throw new DLInvalidTypeException("NodeDef doesn't define a type: " + n);
		}
	}

	private TensorShapeProto getShapeOfNodeDef(final NodeDef n, final boolean input) {
		if (input && n.containsAttr("shape")) {
			return n.getAttrOrThrow("shape").getShape();
		} else if (!input && n.containsAttr("_output_shapes")) {
			return n.getAttrOrThrow("_output_shapes").getList().getShape(0);
		} else {
			return null;
		}
	}

	// ---------------------- Methods on general stuff --------------------

	private DLTensorSpec createTensorSpec(final DLTensorId id, final String name, final TensorShapeProto shapeProto,
			final Class<?> type) {
		// TODO infer dimension order
		DLDimensionOrder dimensionOrder = DLDefaultDimensionOrder.TDHWC;

		// Get the shape and batch size
		if (shapeProto != null) {
			final List<Long> shapeList = new ArrayList<>(
					shapeProto.getDimList().stream().map(d -> d.getSize()).collect(Collectors.toList()));
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
					shape = createPartialTensorShape(shapeList);
				} else {
					// The shape is fixed
					shape = createFixedTensorShape(shapeList);
				}
				if (batchSize > 0) {
					return new DLDefaultTensorSpec(id, name, batchSize, shape, type, dimensionOrder);
				} else {
					return new DLDefaultTensorSpec(id, name, shape, type, dimensionOrder);
				}
			}
		}

		// Getting the shape and batch size wasn't successful
		return new DLDefaultTensorSpec(id, name, type, dimensionOrder);
	}

	private DLTensorShape createPartialTensorShape(final List<Long> shapeList) {
		final OptionalLong[] dims = shapeList.stream().map(l -> l > 0 ? OptionalLong.of(l) : OptionalLong.empty())
				.toArray(OptionalLong[]::new);
		return new DLDefaultPartialTensorShape(dims);
	}

	private DLTensorShape createFixedTensorShape(final List<Long> shapeList) {
		final long[] dims = shapeList.stream().mapToLong(Long::longValue).toArray();
		return new DLDefaultFixedTensorShape(dims);
	}

	private boolean canBeInputOrOutput(final DataType t) {
		try {
			return !getClassForType(t).equals(String.class);
		} catch (final DLInvalidTypeException e) {
			return false;
		}
	}

	private Class<?> getClassForType(final DataType t) throws DLInvalidTypeException {
		switch (t) {
		case DT_FLOAT:
			return float.class;
		case DT_DOUBLE:
			return double.class;
		case DT_INT32:
			return int.class;
		case DT_INT64:
			return long.class;
		default:
			throw new DLInvalidTypeException("The type " + t + " has no corresponding type in KNIME.");
		}
	}

	private DLDimensionOrder inferDimensionOrderFromGraph(final Collection<MetaGraphDef> metaGraphDefs) {
		// TODO move default somewhere else
		// TODO also look in meta_info_def for ops with data format and default values (but the default value should be
		// the same as ours)
		final DLDimensionOrder defaultDimensionOrder = DLDefaultDimensionOrder.TDHWC;
		return metaGraphDefs.stream().map(m -> m.getGraphDef()).flatMap(g -> g.getNodeList().stream())
				.filter(n -> n.containsAttr("data_format"))
				.map(n -> inferDimensionOrderFromString(n.getAttrOrThrow("data_format").getS().toString()))
				.filter(o -> o.isPresent()).map(o -> o.get()).findFirst().orElse(defaultDimensionOrder);
	}

	private Optional<DLDimensionOrder> inferDimensionOrderFromString(final String dimOrder) {
		// TODO make more powerful
		if (dimOrder.equals("NDHWC") || dimOrder.equals("NHWC") || dimOrder.equals("NWC")) {
			return Optional.of(DLDefaultDimensionOrder.TDHWC);
		} else if (dimOrder.equals("NCDHW") || dimOrder.equals("NCHW") || dimOrder.equals("NCW")) {
			return Optional.of(DLDefaultDimensionOrder.TCDHW);
		} else {
			return Optional.empty();
		}
	}
}
