#!groovy
def BN = (BRANCH_NAME == 'master' || BRANCH_NAME.startsWith('releases/')) ? BRANCH_NAME : 'releases/2022-06'

library "knime-pipeline@$BN"

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
                'knime-tensorflow', 'knime-deeplearning', 'knime-python', 'knime-filehandling', 'knime-jfreechart', 'knime-distance', 'knime-kerberos'
            ]
        ]
    )

    stage('Sonarqube analysis') {
        env.lastStage = env.STAGE_NAME
        workflowTests.runSonar()
    }
} catch (ex) {
    currentBuild.result = 'FAILURE'
    throw ex
} finally {
    notifications.notifyBuild(currentBuild.result);
}
/* vim: set shiftwidth=4 expandtab smarttab: */
