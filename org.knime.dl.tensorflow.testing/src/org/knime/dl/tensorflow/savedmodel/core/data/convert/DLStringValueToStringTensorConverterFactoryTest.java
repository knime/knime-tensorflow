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
package org.knime.dl.tensorflow.savedmodel.core.data.convert;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;
import java.util.OptionalLong;
import java.util.stream.Collectors;

import org.junit.Test;
import org.knime.core.data.DataColumnSpec;
import org.knime.core.data.StringValue;
import org.knime.core.data.def.StringCell;
import org.knime.dl.core.DLDefaultFixedTensorShape;
import org.knime.dl.core.DLDefaultTensor;
import org.knime.dl.core.DLTensor;
import org.knime.dl.core.data.convert.DLDataValueToTensorConverter;
import org.knime.dl.tensorflow.savedmodel.core.data.DLWritableStringBuffer;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorStringBuffer;
import org.knime.dl.tensorflow.testing.TFTestUtil;

/**
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public class DLStringValueToStringTensorConverterFactoryTest {

	private static DLStringValueToStringTensorConverterFactory createFactory() {
		return new DLStringValueToStringTensorConverterFactory();
	}
	
	@Test
	public void testGetName() throws Exception {
		assertEquals("String", createFactory().getName());
	}
	
	@Test
	public void testGetSourceType() throws Exception {
		assertEquals(StringValue.class, createFactory().getSourceType());
	}
	
	@Test
	public void testGetBufferType() throws Exception {
		assertEquals(DLWritableStringBuffer.class, createFactory().getBufferType());
	}
	
	@Test
	public void testGetDestCount() throws Exception {
		DataColumnSpec colSpec = null;
		List<DataColumnSpec> spec = Arrays.asList(colSpec, colSpec);
		assertEquals(OptionalLong.of(2l), createFactory().getDestCount(spec));
	}
	
	@Test
	public void testCreateConverter() throws Exception {
		try (TFTensorStringBuffer buffer = new TFTensorStringBuffer(new long[] {1l, 10l});
		DLTensor<DLWritableStringBuffer> tensor = new DLDefaultTensor<DLWritableStringBuffer>(
				TFTestUtil.createSpec(new DLDefaultFixedTensorShape(new long[] {10l})),
				buffer, 10l)) {
			String[] values = new String[] {"deep", "learning", "in", "knime"};
			List<StringValue> input = Arrays.stream(values).map(StringCell::new)
					.collect(Collectors.toList());
			DLDataValueToTensorConverter<StringValue, DLWritableStringBuffer> converter = createFactory().createConverter();
			converter.convert(input, tensor);
			assertEquals(values.length, buffer.size());
			for (int i = 0; i < buffer.size(); i++) {
				assertEquals(values[i], buffer.readNext());
			}
		}
	}
}
