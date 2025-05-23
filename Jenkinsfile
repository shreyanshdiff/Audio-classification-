pipeline {
    agent any 
    environment {
        PYTHON = 'python3'
    }
    triggers {
        cron('H 21 * * 4') 
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Setup') {
            steps {
                script {
                    sh "${PYTHON} -m pip install --upgrade pip"
                    sh "${PYTHON} -m pip install -r requirements.txt"
                }
            }
        }
        stage('Train Model') {
            steps {
                script {
                    sh "${PYTHON} model.py > training_logs.txt"
                }
            }
        }
        stage('Archive Results') {
            steps {
                archiveArtifacts artifacts: 'audio_classification_model.h5', fingerprint: true
                archiveArtifacts artifacts: 'label_encoder.pkl', fingerprint: true
                archiveArtifacts artifacts: 'confusion_matrix.png', fingerprint: true
                archiveArtifacts artifacts: 'training_history.png', fingerprint: true
                archiveArtifacts artifacts: 'training_logs.txt', fingerprint: true
            }
        }
    }
    post {
        always {
            emailext (
                subject: 'Weekly Model Training Results - ${BUILD_STATUS}',
                body: '''Build: ${BUILD_URL}
                
                Training Logs:
                ${FILE,path="training_logs.txt"}
                
                Artifacts:
                - Model: ${BUILD_URL}artifact/audio_classification_model.h5
                - Label Encoder: ${BUILD_URL}artifact/label_encoder.pkl
                - Confusion Matrix: ${BUILD_URL}artifact/confusion_matrix.png
                - Training History: ${BUILD_URL}artifact/training_history.png''',
                to: 'shreyansh1702@gmail.com',
                attachLog: true
            )
            cleanWs()
        }
    }
}