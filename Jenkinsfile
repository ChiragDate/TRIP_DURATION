pipeline {
    agent any
    
    environment {
        // Define environment variables
        PYTHON_VERSION = '3.12.3'
        DVC_MODELS_DIR = "/home/chirag/Projects/TRIP-DURATION-MODELS"
        VENV_PATH = "${WORKSPACE}/trip_duration_venv"
        WORK = "${WORKSPACE}"
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
                '''
            }
        }
        
        stage('Pull Models') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    dvc --version
                    
                    # Pull models from DVC tracking
                    dvc pull --force
                '''
            }
        }
        
        stage('Build Features and store csv') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    
                    # Run your code that uses the models
                    python3 src/features/build_features.py ${WORK}/data/raw/
                '''
            }
        }
            stage('Train Model') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate

                    if [ -f ""${WORK}/models/model.joblib"" ]; then
                        echo "Model already exists at $MODEL_FILE - skipping training"
                    else
                        echo "Model not found - starting training process"
                        python3 ${WORK}/src/models/train_model.py ${WORK}/data/processed/train.csv ${WORK}/models
                    fi
                '''
            }
        }
        stage('Run API Service') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate

                    # Start FastAPI using uvicorn
                    nohup uvicorn src.service:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &

                    echo $! > api_pid.txt

                    # Wait for API to start
                    sleep 5

                    # Check if API is running
                    if ! curl -s http://localhost:8000/health; then
                    echo "FASTAPI is not running"
                    exit 1
                    fi
                '''
            }
        }

        
        stage('Test API') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    
                    # Run API tests
                    python -m pytest ${WORK}/tests/test_api.py -v
                    
                    # Test with a sample prediction request
                    curl -s -X POST http://localhost:8000/predict \
                    -H "Content-Type: application/json" \
                    --data-binary "@/var/lib/jenkins/workspace/TRIP_DURATION/tests/sample_request.json" \
                    | tee prediction_output.json


                    
                    # Validate the prediction output format
                    python -c "import json; data = json.load(open('prediction_output.json')); assert 'prediction' in data, 'Missing prediction key in response'"
                    
                    # Cleanup API process
                    kill $(cat api_pid.txt)
                    rm api_pid.txt
                '''
            }
        }
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