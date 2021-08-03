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
package org.knime.dl.tensorflow.core.convert;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.apache.commons.lang3.NotImplementedException;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.junit.Test;
import org.knime.core.data.filestore.FileStore;
import org.knime.dl.base.portobjects.DLNetworkPortObject;
import org.knime.dl.core.DLAbstractNetwork;
import org.knime.dl.core.DLAbstractNetworkSpec;
import org.knime.dl.core.DLCancelable;
import org.knime.dl.core.DLNetwork;
import org.knime.dl.core.DLNetworkSpec;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.core.training.DLTrainingConfig;
import org.knime.dl.tensorflow.core.TFNetwork;
import org.knime.dl.tensorflow.core.TFNetworkSpec;

/**
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TFNetworkConverterRegistryTest {

	private static final TFNetworkConverterRegistry CONVERTER_REGISTRY = TFNetworkConverterRegistry.getInstance();

	/**
	 * Tests the {@link TFNetworkConverterRegistry}.
	 *
	 * @throws Exception if the test fails
	 */
	@Test
	public void testNetworkConverterRegistry() throws Exception {
		TFNetworkConverter converter = CONVERTER_REGISTRY.getConverter(DummyNetwork.class);
		assertNotNull(converter);
		assertTrue(converter instanceof DummyNetworkConverter);

		converter = CONVERTER_REGISTRY.getConverter(DummyNetwork2.class);
		assertNull(converter);
	}

	/**
	 * Dummy network converter.
	 */
	public static class DummyNetworkConverter extends TFAbstractNetworkConverter<Void, DummyNetwork> {

		/**
		 * Creates instance of dummy network converter.
		 */
		public DummyNetworkConverter() {
			super(DummyNetwork.class, TFNetwork.class);
		}

        @Override
        protected DummyNetwork extractNetworkFromPortObject(final Void noContext,
            final DLNetworkPortObject networkPortObject) throws DLNetworkConversionException {
            throw new NotImplementedException("Should not be called");
        }

		@Override
        protected TFNetwork convertNetworkInternal(final Void noContext, final DummyNetwork network,
            final FileStore fileStore, final DLCancelable cancelable) throws DLNetworkConversionException {
			throw new NotImplementedException("Should not be called");
		}

		@Override
		public void checkSpec(final DLNetworkSpec spec) throws DLNetworkConversionException {
			// nothing to do
		}
	}

	/**
	 * Dummy network type for testing the network converter registry.
	 */
	public static class DummyNetwork extends DLAbstractNetwork<DummyNetworkSpec> {

		/**
		 * @param spec
		 */
		protected DummyNetwork(final DummyNetworkSpec spec) {
			super(spec);
		}

		@Override
		protected void hashCodeInternal(final HashCodeBuilder b) {
			// nothing to do
		}

		@Override
		protected boolean equalsInternal(final DLNetwork other) {
			return true;
		}

	}

	/**
	 * Dummy network spec for testing the network converter registry.
	 */
	public static class DummyNetworkSpec extends DLAbstractNetworkSpec<DLTrainingConfig> {

		private static final long serialVersionUID = 1L;

		/**
		 * @param inputSpecs
		 * @param hiddenOutputSpecs
		 * @param outputSpecs
		 */
		protected DummyNetworkSpec(final DLTensorSpec[] inputSpecs, final DLTensorSpec[] hiddenOutputSpecs,
				final DLTensorSpec[] outputSpecs) {
			super(TFNetworkSpec.getTFBundleVersion(), inputSpecs, hiddenOutputSpecs, outputSpecs);
		}

		@Override
		protected void hashCodeInternal(final HashCodeBuilder b) {
			// nothing to do
		}

		@Override
		protected boolean equalsInternal(final DLNetworkSpec other) {
			return true;
		}

	}

	/**
	 * Dummy network type for testing the network converter registry.
	 */
	public class DummyNetwork2 extends DLAbstractNetwork<DummyNetworkSpec> {

		/**
		 * @param spec
		 */
		protected DummyNetwork2(final DummyNetworkSpec spec) {
			super(spec);
		}

		@Override
		protected void hashCodeInternal(final HashCodeBuilder b) {
			// nothing to do
		}

		@Override
		protected boolean equalsInternal(final DLNetwork other) {
			return true;
		}

	}
}
