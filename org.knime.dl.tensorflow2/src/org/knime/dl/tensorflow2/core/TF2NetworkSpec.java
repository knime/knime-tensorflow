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
 * History
 *   Apr 27, 2020 (benjamin): created
 */
package org.knime.dl.tensorflow2.core;

import static com.google.common.base.Preconditions.checkNotNull;

import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectStreamException;
import java.io.Serializable;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.knime.core.util.Version;
import org.knime.dl.core.DLAbstractNetworkSpec2;
import org.knime.dl.core.DLNetworkSpec;
import org.knime.dl.core.DLTensorSpec;
import org.knime.dl.python.core.DLPythonNetworkSpec;
import org.knime.dl.util.DLUtils;

/**
 * The specs of a {@link TF2Network}.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public class TF2NetworkSpec extends DLAbstractNetworkSpec2<TF2TrainingConfig> implements DLPythonNetworkSpec {

    // TODO add training config (also to serialization)

    private final Version m_pythonVersion;

    private final Version m_tfVersion;

    /**
     * Create the specification for a {@link TF2Network}.
     *
     * The Python and TensorFlow version are saved because they can be useful if we discover problems with certain
     * versions.
     *
     * @param pythonVersion The Python version used to create the network
     * @param tfVersion The TensorFlow version used to create the network
     * @param inputSpecs
     * @param hiddenOutputSpecs
     * @param outputSpecs
     */
    public TF2NetworkSpec(final Version pythonVersion, final Version tfVersion, final DLTensorSpec[] inputSpecs,
        final DLTensorSpec[] hiddenOutputSpecs, final DLTensorSpec[] outputSpecs) {
        super(getTFBundleVersion(), inputSpecs, hiddenOutputSpecs, outputSpecs);
        m_pythonVersion = checkNotNull(pythonVersion);
        m_tfVersion = checkNotNull(tfVersion);
    }

    /** Constructor for deserialization */
    private TF2NetworkSpec(final Version bundleVersion, final Version pythonVersion, final Version tfVersion,
        final DLTensorSpec[] inputSpecs, final DLTensorSpec[] hiddenOutputSpecs, final DLTensorSpec[] outputSpecs) {
        super(bundleVersion, inputSpecs, hiddenOutputSpecs, outputSpecs);
        m_pythonVersion = pythonVersion;
        m_tfVersion = tfVersion;
    }

    @Override
    protected void hashCodeInternal(final HashCodeBuilder b) {
        // Nothing to add to the hash code
    }

    @Override
    protected boolean equalsInternal(final DLNetworkSpec other) {
        // Everything checked in the super class
        // m_tfVersion and m_pythonVersion are not considered
        return true;
    }

    @Override
    public Version getPythonVersion() {
        return m_pythonVersion;
    }

    /**
     * Returns the version of the TensorFlow library this network has been created with. Please note that the version
     * should only be used for version management (e.g. resolving backward compatibility issues). As such, it is not
     * considered by {@link #hashCode()} and {@link #equals(Object)}.
     *
     * @return the TensorFlow version
     */
    public Version getTFVersion() {
        return m_tfVersion;
    }

    /** Use the serialization proxy for serialization */
    private Object writeReplace() throws ObjectStreamException {
        return new SerializationProxy(this);
    }

    /** Prevent deserialization of this class. The serialization proxy has to be used. */
    @SuppressWarnings({"static-method", "unused"})
    private void readObject(final ObjectInputStream stream) throws IOException, ClassNotFoundException {
        throw new InvalidObjectException("Use Serialization Proxy instead.");
    }

    /** The SerializationProxy used for serialization and deserialization. */
    private static final class SerializationProxy implements Serializable {

        private static final long serialVersionUID = 1L;

        private DLAbstractNetworkSpec2.SerializationProxy m_parentProxy;

        private Version m_pythonVersion;

        private Version m_tfVersion;

        private SerializationProxy(final TF2NetworkSpec obj) {
            m_parentProxy = new DLAbstractNetworkSpec2.SerializationProxy(obj);
            m_pythonVersion = obj.m_pythonVersion;
            m_tfVersion = obj.m_tfVersion;
        }

        private Object readResolve() throws ObjectStreamException {
            return new TF2NetworkSpec(new Version(m_parentProxy.m_bundleVersionString), m_pythonVersion, m_tfVersion,
                m_parentProxy.m_inputSpec, m_parentProxy.m_hiddenOutputSpec, m_parentProxy.m_outputSpec);
        }
    }

    /** @return the version of the KNIME Deep Learning Tensorflow bundle */
    private static Version getTFBundleVersion() {
        return DLUtils.Misc.getVersionOfSameBundle(TF2NetworkSpec.class);
    }
}
