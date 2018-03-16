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

/**
 * @author Adrian Nembach, KNIME GmbH, Konstanz, Germany
 */
final class DLRankSixBytesBuffer extends DLAbstractBytesBuffer<byte[][][][][][][]> {

	/**
	 * @param shape
	 */
	protected DLRankSixBytesBuffer(long[] shape) {
		super(shape);
	}

	@Override
	protected byte[] retrieveFromStorage(int[] position) {
		assert position.length == 6;
		return m_storage[position[0]][position[1]][position[2]][position[3]][position[4]][position[5]];
	}

	@Override
	protected void placeInStorage(byte[] value, int[] position) {
		assert position.length == 6;
		m_storage[position[0]][position[1]][position[2]][position[3]][position[4]][position[5]] = value;
	}

	@Override
	protected long getLength(byte[][][][][][][] storage) {
		long dim1 = storage.length;
		long dim2 = storage[0].length;
		long dim3 = storage[0][0].length;
		long dim4 = storage[0][0][0].length;
		long dim5 = storage[0][0][0][0].length;
		long dim6 = storage[0][0][0][0][0].length;
		return dim1 * dim2 * dim3 * dim4 * dim5 * dim6;
	}

	@Override
	protected byte[][][][][][][] createEmptySubArray(int length) {
		return new byte[length][][][][][][];
	}

	@Override
	protected byte[][][][][][][] createStorage(int[] shape) {
		if (shape.length != 6) {
			throw new IllegalArgumentException(
					"Invalid shape. Can't create a DLRankSixBytesBuffer from a rank " + shape.length + " shape.");
		}
		return new byte[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][];
	}

}
