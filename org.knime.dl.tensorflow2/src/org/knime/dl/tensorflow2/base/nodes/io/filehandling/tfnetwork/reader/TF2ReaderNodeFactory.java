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
 *   May 22, 2020 (benjamin): created
 */
package org.knime.dl.tensorflow2.base.nodes.io.filehandling.tfnetwork.reader;

import org.knime.core.node.context.NodeCreationConfiguration;
import org.knime.core.node.port.PortType;
import org.knime.dl.tensorflow2.base.portobjects.TF2NetworkPortObject;
import org.knime.filehandling.core.node.portobject.SelectionMode;
import org.knime.filehandling.core.node.portobject.reader.PortObjectReaderNodeConfig;
import org.knime.filehandling.core.node.portobject.reader.PortObjectReaderNodeDialog;
import org.knime.filehandling.core.node.portobject.reader.PortObjectReaderNodeFactory;

/**
 * Node factory of the TensorFlow 2 network reader node.
 *
 * @author Benjamin Wilhelm, KNIME GmbH, Konstanz, Germany
 */
public final class TF2ReaderNodeFactory
    extends PortObjectReaderNodeFactory<TF2ReaderNodeModel, PortObjectReaderNodeDialog<PortObjectReaderNodeConfig>> {

    /** History Id for the file chooser */
    private static final String HISTORY_ID = "tf2_network_reader_writer";

    /** File extension/suffix */
    private static final String[] FILE_SUFFIX = new String[]{".zip", ".h5"};

    @Override
    protected PortObjectReaderNodeDialog<PortObjectReaderNodeConfig>
        createDialog(final NodeCreationConfiguration creationConfig) {
        // TODO(filehandling) The dialog does show an error if a directory is selected
        return new PortObjectReaderNodeDialog<>(getConfig(creationConfig), HISTORY_ID, SelectionMode.FILE_AND_FOLDER);
    }

    @Override
    protected TF2ReaderNodeModel createNodeModel(final NodeCreationConfiguration creationConfig) {
        return new TF2ReaderNodeModel(creationConfig, getConfig(creationConfig));
    }

    @Override
    protected PortType getOutputPortType() {
        return TF2NetworkPortObject.TYPE;
    }

    /** The reader configuration */
    private static PortObjectReaderNodeConfig getConfig(final NodeCreationConfiguration creationConfig) {
        return new PortObjectReaderNodeConfig(creationConfig, FILE_SUFFIX);
    }
}