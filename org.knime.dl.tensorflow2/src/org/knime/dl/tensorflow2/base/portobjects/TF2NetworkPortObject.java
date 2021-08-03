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
package org.knime.dl.tensorflow2.base.portobjects;

import java.io.File;
import java.io.IOException;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.knime.core.data.filestore.FileStore;
import org.knime.core.node.CanceledExecutionException;
import org.knime.core.node.ExecutionMonitor;
import org.knime.core.node.port.PortObjectSpec;
import org.knime.core.node.port.PortObjectZipInputStream;
import org.knime.core.node.port.PortObjectZipOutputStream;
import org.knime.core.node.port.PortType;
import org.knime.core.node.port.PortTypeRegistry;
import org.knime.dl.base.portobjects.DLAbstractNetworkPortObject;
import org.knime.dl.base.portobjects.DLNetworkPortObject;
import org.knime.dl.core.DLInvalidSourceException;
import org.knime.dl.core.DLNetworkFileStoreLocation;
import org.knime.dl.python.core.DLPythonNetworkPortObject;
import org.knime.dl.tensorflow2.core.TF2Network;
import org.knime.python2.PythonCommand;

/**
 * TensorFlow 2 implementation of a deep learning {@link DLNetworkPortObject}.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TF2NetworkPortObject extends DLAbstractNetworkPortObject<TF2Network, TF2NetworkPortObjectSpec>
    implements DLPythonNetworkPortObject<TF2Network> {

    /**
     * The TensorFlow deep learning network port type
     */
    @SuppressWarnings("hiding")
    public static final PortType TYPE = PortTypeRegistry.getInstance().getPortType(TF2NetworkPortObject.class);

    /**
     * Creates a new TensorFlow deep learning network port object. The given network must be stored in the given file
     * store.
     *
     * @param network the TensorFlow deep learning network to store
     * @param fileStore the file store in which the network is stored
     * @throws IOException if failed to store the network
     */
    public TF2NetworkPortObject(final TF2Network network, final FileStore fileStore) throws IOException {
        super(network, new TF2NetworkPortObjectSpec(network.getSpec(), network.getClass()), fileStore);
        checkNetworkLocation(network, fileStore);
    }

    /**
     * Empty framework constructor. Must not be called by client code.
     */
    public TF2NetworkPortObject() {
        super();
    }

    private static void checkNetworkLocation(final TF2Network network, final FileStore fileStore) {
        final File networkSource = new File(network.getSource().getURI());
        final File fileStoreLoc = fileStore.getFile();
        if (!networkSource.equals(fileStoreLoc)) {
            throw new IllegalStateException("The TF2 network is not saved in the given FileStore. Network source: \""
                + networkSource.getAbsolutePath() + "\", File store: \"" + fileStoreLoc.getAbsolutePath()
                + "\". This is an implementation error.");
        }
    }

    @Override
    protected void flushToFileStoreInternal(final TF2Network network, final FileStore fileStore) throws IOException {
        // Nothing to do: The network is already saved in the file store (Checked in the constructor)
    }

    @Override
    public String getModelName() {
        return "TensorFlow 2 Deep Learning Network";
    }

    @Override
    protected void hashCodeInternal(final HashCodeBuilder b) {
        // Nothing to do
    }

    @Override
    protected boolean equalsInternal(final DLNetworkPortObject other) {
        // Nothing to check (Checks are done by super class)
        return true;
    }

    /**
     * TensorFlow 2 networks are always materialized, so simply delegate to the base implementation.
     * <P>
     * {@inheritDoc}
     */
    @Override
    public TF2Network getNetwork(final PythonCommand command) throws DLInvalidSourceException, IOException {
        return super.getNetwork();
    }

    @Override
    protected TF2Network getNetworkInternal(final TF2NetworkPortObjectSpec spec)
        throws DLInvalidSourceException, IOException {
        return new TF2Network(spec.getNetworkSpec(), new DLNetworkFileStoreLocation(getFileStore(0)));
    }

    /**
     * Serializer for {@link TF2NetworkPortObject}
     */
    public static final class Serializer extends PortObjectSerializer<TF2NetworkPortObject> {

        @Override
        public void savePortObject(final TF2NetworkPortObject portObject, final PortObjectZipOutputStream out,
            final ExecutionMonitor exec) throws IOException, CanceledExecutionException {
            // Nothing to write. The network is defined by its specs and the file store
        }

        @Override
        public TF2NetworkPortObject loadPortObject(final PortObjectZipInputStream in, final PortObjectSpec spec,
            final ExecutionMonitor exec) throws IOException, CanceledExecutionException {
            final TF2NetworkPortObject portObject = new TF2NetworkPortObject();
            portObject.m_spec = (TF2NetworkPortObjectSpec)spec;
            return portObject;
        }

    }

}
