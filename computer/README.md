## Setup Scripts with Anaconda

### 1. Install Anaconda / Miniconda
Download and install [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

### 2. Create a new environment
```bash
conda create -n bluebagedgeai python=3.11
conda activate bluebagedgeai
```

### 3. Install YOLO

Follow the steps here:

https://docs.ultralytics.com/quickstart/

### 4. Install other dependencies
```bash
pip install redis kagglehub
```

### 5. Setup n8n, Ollama, and Redis

- Follow the steps here:  
  https://github.com/n8n-io/self-hosted-ai-starter-kit/tree/main

- Import the workflow:  
  `computer/n8n_workflow/Garbage Recycling.json`  

  After the import, it should match the image:  
  ![Workflow](https://github.com/scardoso-lu/BlueBagEdgeAI/blob/main/computer/n8n_workflow/workflow.png)

- Check the Service Credentials:  
  They should match both images:

  **Ollama service**  
  ![Ollama Service](https://github.com/scardoso-lu/BlueBagEdgeAI/blob/main/computer/n8n_workflow/ollama_service.png)

  **Redis service**  
  ![Redis Service](https://github.com/scardoso-lu/BlueBagEdgeAI/blob/main/computer/n8n_workflow/redis.png)

- Activate the workflow:  
  ![Workflow Running](https://github.com/scardoso-lu/BlueBagEdgeAI/blob/main/computer/n8n_workflow/running.png)

### 6. Run the python scripts on different terminals

```bash
python detect.py
python light.py
```
