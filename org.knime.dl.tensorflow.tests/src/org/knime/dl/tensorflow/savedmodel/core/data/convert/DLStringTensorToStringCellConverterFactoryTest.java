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

import java.util.OptionalLong;

import org.junit.Test;
import org.knime.core.data.DataType;
import org.knime.core.data.def.StringCell;
import org.knime.dl.core.DLDefaultFixedTensorShape;
import org.knime.dl.core.DLDefaultTensor;
import org.knime.dl.core.DLTensor;
import org.knime.dl.core.data.DLReadableStringBuffer;
import org.knime.dl.core.data.convert.DLTensorToDataCellConverter;
import org.knime.dl.tensorflow.savedmodel.core.data.TFTensorStringBuffer;
import org.knime.dl.tensorflow.testing.TFTestUtil;

/**
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public class DLStringTensorToStringCellConverterFactoryTest {

	private static DLStringTensorToStringCellConverterFactory createFactory() {
		return new DLStringTensorToStringCellConverterFactory();
	}

	@Test
	public void testGetDestCount() throws Exception {
		DLStringTensorToStringCellConverterFactory factory = createFactory();
		assertEquals(OptionalLong.of(10l),
				factory.getDestCount(TFTestUtil.createSpec(new DLDefaultFixedTensorShape(new long[] { 10 }))));
	}

	@Test
	public void testGetDestType() throws Exception {
		DLStringTensorToStringCellConverterFactory factory = createFactory();
		assertEquals(DataType.getType(StringCell.class), factory.getDestType());
	}

	@Test
	public void testGetBufferType() throws Exception {
		DLStringTensorToStringCellConverterFactory factory = createFactory();
		assertEquals(DLReadableStringBuffer.class, factory.getBufferType());
	}

	@Test
	public void testGetName() throws Exception {
		DLStringTensorToStringCellConverterFactory factory = createFactory();
		assertEquals("String", factory.getName());
	}

	@Test
	public void testCreateConverter() throws Exception {
		DLStringTensorToStringCellConverterFactory factory = createFactory();
		String value = "knime";
		try (TFTensorStringBuffer buffer = new TFTensorStringBuffer(new long[] { 1l, 10l });
				DLTensor<DLReadableStringBuffer> tensor = new DLDefaultTensor<DLReadableStringBuffer>(
						TFTestUtil.createSpec(new DLDefaultFixedTensorShape(new long[] { 10l })), buffer, 10l)) {
			TFTestUtil.fillBufferWithValue(buffer, value);
			DLTensorToDataCellConverter<DLReadableStringBuffer, StringCell> converter = factory.createConverter();
			StringCell[] output = new StringCell[10];
			converter.convert(tensor, output, null);
			for (int i = 0; i < output.length; i++) {
				assertEquals(value, output[i].getStringValue());
			}
		}
	}
}
