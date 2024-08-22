#!groovy
def BN = (BRANCH_NAME == 'master' || BRANCH_NAME.startsWith('releases/')) ? BRANCH_NAME : 'releases/2024-12'

library "knime-pipeline@todo/DEVOPS-2151-workflow-tests-default-mac-os-arm"

properties([
    pipelineTriggers([
        upstream('knime-deeplearning/' + env.BRANCH_NAME.replaceAll('/', '%2F'))
    ]),
    parameters(workflowTests.getConfigurationsAsParameters()),
    buildDiscarder(logRotator(numToKeepStr: '5')),
    disableConcurrentBuilds()
])

try {
    knimetools.defaultTychoBuild('org.knime.update.tensorflow')

    workflowTests.runTests(
        dependencies: [
            repositories: [
                'knime-tensorflow', 'knime-deeplearning', 'knime-python-legacy', 'knime-conda',
                'knime-filehandling', 'knime-jfreechart', 'knime-distance', 'knime-kerberos'
            ]
        ],
        configurations: ['MacOS_12_M1_knime420', 'MacOS_13_M1_knime421', 'MacOS_14_M1_knime494'],
    )

    // stage('Sonarqube analysis') {
    //     env.lastStage = env.STAGE_NAME
    //     workflowTests.runSonar()
    // }
} catch (ex) {
    currentBuild.result = 'FAILURE'
    throw ex
} finally {
    notifications.notifyBuild(currentBuild.result);
}
/* vim: set shiftwidth=4 expandtab smarttab: */
