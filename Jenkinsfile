pipeline {
    agent any
    
    environment {
        // Define environment variables
        PYTHON_VERSION = '3.10.1'
        DVC_MODELS_DIR = "/home/chirag/Projects/TRIP-DURATION-MODELS"
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                // Create and activate Python virtual environment
                sh '''
                python3 -m venv /tmp/trip_duration_venv
                . /tmp/trip_duration_venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt

                pip install dvc
                '''
            }
        }
        
        stage('Initialize DVC') {
            steps {
                sh '''
                    . /tmp/trip_duration_venv/bin/activate
                    # Initialize DVC if not already done
                    if [ ! -d .dvc ]; then
                        dvc init
                    fi
                    
                    # Configure DVC to use local directory for models
                    dvc config cache.dir ${DVC_MODELS_DIR}
                '''
            }
        }
        
        stage('Pull Models') {
            steps {
                sh '''
                    . /tmp/trip_duration_venv/bin/activate
                    
                    # Pull models from DVC tracking
                    dvc pull
                '''
            }
        }
        
        // stage('Run Code with Models') {
        //     steps {
        //         sh '''
        //             . /tmp/trip_duration_venv/bin/activate
        //             # Run your code that uses the models
        //             python src/run_model.py
        //         '''
        //     }
        // }
        
        // stage('Track Model Changes') {
        //     steps {
        //         sh '''
        //             . venv/bin/activate
        //             # Add any new model outputs to DVC tracking
        //             dvc add models/new_output_model.pkl
                    
        //             # Commit DVC changes
        //             git add .dvc/config models/*.dvc
        //             git commit -m "Update model tracking" || echo "No changes to commit"
        //         '''
        //     }
        // }
    // }
    
    // post {
    //     always {
    //         // Cleanup steps
    //         cleanWs()
    //     }
    // }
}
}
