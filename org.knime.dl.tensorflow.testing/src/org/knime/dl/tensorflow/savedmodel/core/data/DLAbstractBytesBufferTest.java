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
package org.knime.dl.tensorflow.savedmodel.core.data;

import static org.junit.Assert.*;

import java.nio.BufferOverflowException;
import java.nio.BufferUnderflowException;
import java.util.Arrays;

import org.junit.Test;

/**
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
public class DLAbstractBytesBufferTest {

	private static byte[][] createRange(int length) {
		byte[][] expected = new byte[length][1];
		for (byte i = 0; i < length; i++) {
			expected[i][0] = i;
		}
		return expected;
	}

	@Test
	public void testGetStorageForTensorCreation() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { 10l })) {
			byte[][] expected = createRange(10);
			buffer.putAll(expected);
			assertArrayEquals(expected, buffer.getStorageForTensorCreation(10l));
			buffer.resetRead();
			assertArrayEquals(Arrays.copyOf(expected, 5), buffer.getStorageForTensorCreation(5l));
		}
	}

	@Test
	public void testSetStorage() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { 10l })) {
			byte[][] storage = createRange(10);
			buffer.setStorage(storage, 10l);
			assertArrayEquals(storage, buffer.getStorageForReading(0, 10));
		}
	}

	@Test(expected = IllegalArgumentException.class)
	public void testSetStorageFailsOnIncompatibleCapacity() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { 10l })) {
			byte[][] storage = createRange(5);
			buffer.setStorage(storage, 10l);
		}
	}

	@Test(expected = BufferOverflowException.class)
	public void testPutFailsOnOverflow() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { 1l })) {
			buffer.put(new byte[1]);
			buffer.put(new byte[1]);
		}
	}

	@Test(expected = BufferOverflowException.class)
	public void testPutAllFailsOnOverflow() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { 10l })) {
			byte[][] tooLargeArray = createRange(11);
			buffer.putAll(tooLargeArray);
		}
	}

	@Test(expected = BufferUnderflowException.class)
	public void testReadNextFailsOnUnderflow() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { 10l })) {
			buffer.readNext();
		}
	}

	@Test(expected = IllegalArgumentException.class)
	public void testConstructorFailsOnTooLargeDimension() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { Long.MAX_VALUE })) {
		}
	}

	@Test(expected = IllegalArgumentException.class)
	public void testZeroPadFailsOnLengthSmallerOne() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { 10l })) {
			buffer.zeroPad(0);
		}
	}

	@Test(expected = BufferOverflowException.class)
	public void testZeroPadFailsOnOverflow() throws Exception {
		try (DLRankOneBytesBuffer buffer = new DLRankOneBytesBuffer(new long[] { 10l })) {
			buffer.zeroPad(11);
		}
	}
}
