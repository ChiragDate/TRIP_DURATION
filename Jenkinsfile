pipeline {
    agent any
    
    environment {
        // Define environment variables
        PYTHON_VERSION = '3.10.1'
        DVC_MODELS_DIR = "/home/chirag/Projects/TRIP-DURATION-MODELS"
        VENV_PATH = "${WORKSPACE}/trip_duration_venv"
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                sh '''
                    # Create virtual environment in workspace for proper permissions
                    python3 -m venv ${VENV_PATH}
                    
                    # Make sure activation script is executable
                    chmod +x ${VENV_PATH}/bin/activate
                    
                    # Activate virtual environment and install dependencies
                    . ${VENV_PATH}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                    pip install dvc
                '''
            }
        }
        
        stage('Initialize DVC') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    
                    # Initialize DVC if not already done
                    if [ ! -d .dvc ]; then
                        dvc init
                    fi
                    
                    # Configure DVC to use local directory for models
                    mkdir -p ${DVC_MODELS_DIR}
                    dvc config cache.dir ${DVC_MODELS_DIR}
                    
                    # Make sure Jenkins has write permissions to the models directory
                    chmod -R 755 ${DVC_MODELS_DIR}
                '''
            }
        }
        
        stage('Pull Models') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    dvc --version
                    
                    # Pull models from DVC tracking
                    dvc pull
                '''
            }
        }
        
        // stage('Run Code with Models') {
        //     steps {
        //         sh '''
        //             . ${VENV_PATH}/bin/activate
                    
        //             # Run your code that uses the models
        //             python src/run_model.py
        //         '''
        //     }
        // }
        
        // stage('Track Model Changes') {
        //     steps {
        //         sh '''
        //             . ${VENV_PATH}/bin/activate
                    
        //             # Add any new model outputs to DVC tracking
        //             dvc add models/new_output_model.pkl
                    
        //             # Commit DVC changes
        //             git add .dvc/config models/*.dvc
        //             git commit -m "Update model tracking" || echo "No changes to commit"
        //         '''
        //     }
        // }
    }
    
    // post {
    //     always {
    //         echo "Pipeline completed"
    //         // Uncomment if you want workspace cleanup
    //         // cleanWs()
    //     }
    // }
}