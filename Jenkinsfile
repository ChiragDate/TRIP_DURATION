pipeline {
    agent any
    
    environment {
        // Define environment variables
        PYTHON_VERSION = '3.12.3'
        DVC_MODELS_DIR = "/home/chirag/Projects/TRIP-DURATION-MODELS"
        VENV_PATH = "${WORKSPACE}/trip_duration_venv"
        WORK = "${WORKSPACE}"
        DOCKER_REGISTRY = "docker.io/chiragd02"
        IMAGE_NAME = "trip_duration"
        IMAGE_TAG = "latest"
        DOCKER_CREDENTIALS_ID = "DOCKERHUB"
        KUBECONFIG_ID = "kubeconfig-credentials"
        ANSIBLE_PATH = "${WORKSPACE}/ansible"
        K8S_CLUSTER_NAME = "ml-pipeline-cluster"
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
                    sudo apt-get install -y docker.io
                    sudo systemctl start docker
                    sudo systemctl enable docker
                    pip install dvc ansible
                '''
            }
        }
        
       stage('Check and Create Kubernetes Cluster') {
            steps {
                script {
                    def clusterExists = false

                    // Try to get the cluster info using default kubeconfig
                    try {
                        def status = sh(script: "kubectl cluster-info", returnStatus: true)
                        clusterExists = (status == 0)
                    } catch (Exception e) {
                        echo "Failed to get cluster info: ${e.message}"
                        clusterExists = false
                    }

                    if (!clusterExists) {
                        echo "Kubernetes cluster does not exist. Creating using Ansible..."
                        sh '''
                            cd ${ANSIBLE_PATH}
                            ansible-playbook -i hosts.ini create_k8s_cluster.yml -e "cluster_name=${K8S_CLUSTER_NAME}"

                            # After creating the cluster, store the kubeconfig
                            mkdir -p ~/.kube
                            cp ${ANSIBLE_PATH}/kubeconfig ~/.kube/config
                        '''
                    } else {
                        echo "Kubernetes cluster already exists. Skipping creation."
                    }
                }
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

                    if [ -f "${WORK}/models/model.joblib" ]; then
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
                    sleep 10
                '''
            }
        }

        
        stage('Test API') {
            steps {
                sh '''
                    . ${VENV_PATH}/bin/activate
                    
                    # Run API tests
                    python -m pytest ${WORK}/tests/test_api.py -v
                    
                    
                    # Cleanup API process
                    kill $(cat api_pid.txt)
                    rm api_pid.txt
                '''
            }
        }
        
        stage('Build Docker Image') {
            steps {
                sh '''
                    # Build Docker image
                    docker build -t ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} .
                    
                    # Also tag as latest
                    docker tag ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
                '''
            }
        }
        
        stage('Push Docker Image') {
            steps {
                withDockerRegistry([credentialsId: "${DOCKER_CREDENTIALS_ID}", url: '']) {
                    sh """
                        echo "Pushing image to Docker Hub..."
                        docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:latest
                        echo "Docker push completed successfully"
                    """
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    def deploymentExists = sh(
                        script: "kubectl get deployment trip-duration-api >/dev/null 2>&1",
                        returnStatus: true
                    ) == 0

                    if (deploymentExists) {
                        echo "Deployment 'trip-duration-api' already exists."
                    } else {
                        sh '''
                            kubectl apply -f kubernetes/pvc.yaml
                            kubectl apply -f kubernetes/deployment.yaml
                            kubectl apply -f kubernetes/service.yaml
                            
                            # Wait for deployment to be ready
                            kubectl rollout status deployment/trip-duration-api
                        '''
                    }
                }
            }
        }
    //     stage('Port Forward Service') {
    //     steps {
    //         script {
    //             echo "Port forwarding service to access FastAPI at localhost:8000"
                
    //             // Run port-forward in the background
    //             sh '''
    //                 nohup kubectl port-forward svc/trip-duration-api-service 8000:80
    //             '''
    //         }
    //     }
    // }


        // stage('Setup Monitoring') {
        //     steps {
        //         sh '''
        //             ansible-playbook -i ansible/inventory.ini ansible/setup-elk.yaml
        //         '''

        //     }
        // }

        }

}
    