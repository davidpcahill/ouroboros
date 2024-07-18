#!/usr/bin/env python3
# Ouroboros.py - An AI-driven self-improving experimentation system

import io
import os
import sys
import time
import shutil
import json
import logging
import traceback
import sqlite3
import requests
import configparser
from io import BytesIO
from datetime import datetime
from logging.handlers import RotatingFileHandler

import anthropic
import openai
import git
import docker
import schedule
from tqdm import tqdm
from bs4 import BeautifulSoup
from googleapiclient.discovery import build

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize logging
def setup_logging():
    logger = logging.getLogger('ouroboros')
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = RotatingFileHandler('ouroboros.log', maxBytes=10000000, backupCount=5)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Model token limits
MODEL_MAX_INPUT_TOKENS = {
    # Claude models
    "claude-2.1": 100000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    # OpenAI models
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
}

MODEL_MAX_OUTPUT_TOKENS = {
    # Claude models
    "claude-2.1": 100000,
    "claude-3-haiku-20240307": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-opus-20240229": 4096,
    "claude-3-5-sonnet-20240620": 4096,
    # OpenAI models
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 4096,
    "gpt-4": 8192,
    "gpt-4-32k": 8192,
}

def check_ai_libraries():
    ai_provider = config.get('AI', 'PROVIDER', fallback='claude').lower()
    if ai_provider == 'claude':
        try:
            import anthropic
            required_version = "0.31.2"
            if anthropic.__version__ < required_version:
                logger.error(f"Anthropic version {anthropic.__version__} is below the required version {required_version}. Please upgrade.")
                print(f"Anthropic library version {anthropic.__version__} is outdated. Please upgrade to version {required_version} or later.")
                return False
            return True
        except ImportError:
            logger.error("Anthropic library not found. Please install the library.")
            print("Anthropic library not found. Please install the library.")
            return False
    elif ai_provider == 'openai':
        try:
            import openai
            required_version = "1.35.14"
            if openai.__version__ < required_version:
                logger.error(f"OpenAI version {openai.__version__} is below the required version {required_version}. Please upgrade.")
                print(f"OpenAI library version {openai.__version__} is outdated. Please upgrade to version {required_version} or later.")
                return False
            return True
        except ImportError:
            logger.error("OpenAI library not found. Please install the library.")
            print("OpenAI library not found. Please install the library.")
            return False
    else:
        logger.error(f"Unsupported AI provider: {ai_provider}")
        return False

# Database functions
def init_db():
    conn = None
    try:
        conn = sqlite3.connect('ouroboros.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS experiments
                     (id INTEGER PRIMARY KEY, code TEXT, results TEXT, ai_notes TEXT, dockerfile TEXT, timestamp TEXT)''')
        conn.commit()
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
    finally:
        if conn:
            conn.close()

def get_last_experiment_id():
    conn = None
    try:
        conn = sqlite3.connect('ouroboros.db')
        c = conn.cursor()
        c.execute("SELECT MAX(id) FROM experiments")
        last_id = c.fetchone()[0]
        return last_id or 0
    except sqlite3.Error as e:
        logger.error(f"Error fetching last experiment ID: {e}")
        return 0
    finally:
        if conn:
            conn.close()

# File functions
def read_access():
    try:
        with open('access.txt', 'r') as f:
            access_info = {line.split('=')[0].strip(): line.split('=')[1].strip() for line in f if '=' in line}
        
        ai_provider = config.get('AI', 'PROVIDER', fallback='claude').lower()
        if ai_provider == 'claude':
            if 'ANTHROPIC_API_KEY' not in access_info:
                logger.error("ANTHROPIC_API_KEY not found in access.txt")
        elif ai_provider == 'openai':
            if 'OPENAI_API_KEY' not in access_info:
                logger.error("OPENAI_API_KEY not found in access.txt")
        
        return access_info
    except FileNotFoundError:
        logger.error("access.txt file not found")
        return {}

def check_disk_space(min_space_gb=100):
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    if free_gb < min_space_gb:
        logger.warning(f"Low disk space: {free_gb}GB free. Minimum recommended: {min_space_gb}GB")
    return free_gb

# Git functions
def init_git_repo():
    try:
        exp_dir = os.path.abspath('experiments')
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        
        repo_path = os.path.join(exp_dir, '.git')
        if not os.path.exists(repo_path):
            repo = git.Repo.init(exp_dir)
            logger.info(f"Initialized new Git repository in {exp_dir}")
        else:
            repo = git.Repo(exp_dir)
            logger.info(f"Using existing Git repository in {exp_dir}")
        return repo, exp_dir
    except git.exc.GitCommandError as e:
        logger.error(f"Git repository initialization error: {e}")
        return None, None

def commit_to_git(repo, message, files_to_add=None):
    if repo is None:
        logger.error("Cannot commit: Git repository not initialized")
        return
    try:
        if files_to_add:
            existing_files = [f for f in files_to_add if os.path.exists(f)]
            if existing_files:
                repo.index.add(existing_files)
            else:
                logger.warning(f"No files exist to add for commit: {message}")
                return
        else:
            repo.git.add(A=True)
        repo.index.commit(message)
        logger.info(f"Committed changes to Git: {message}")
    except git.exc.GitCommandError as e:
        logger.error(f"Git commit error: {e}")

# Web functions
def google_search(query, num_results=5):
    if not config.get('Google', 'API_KEY') or not config.get('Google', 'CSE_ID'):
        logger.warning("Google search feature is not configured.")
        return []
    try:
        service = build("customsearch", "v1", developerKey=config.get('Google', 'API_KEY'))
        res = service.cse().list(q=query, cx=config.get('Google', 'CSE_ID'), num=num_results).execute()
        
        results = []
        for item in res.get('items', []):
            results.append({
                'title': item['title'],
                'link': item['link'],
                'snippet': item['snippet']
            })
        logger.info(f"Google search completed for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error in Google search: {str(e)}")
        return []

def load_webpage(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Webpage loaded and parsed: {url}")
        return text[:5000]  # Limit to first 5000 characters
    except Exception as e:
        logger.error(f"Error loading webpage: {str(e)}")
        return f"Error loading webpage: {str(e)}"

# Docker functions
def create_docker_client():
    try:
        client = docker.from_env()
        logger.info("Docker client created successfully")
        return client
    except docker.errors.DockerException as e:
        logger.error(f"Docker client creation error: {e}")
        return None

def run_in_docker(client, container, code, exp_dir):
    try:
        # Execute the code directly in the container
        exit_code, output = container.exec_run(
            cmd=["python", "-c", code],
            demux=True
        )

        result = output[0].decode() if output[0] else ""
        error_output = output[1].decode() if output[1] else ""

        if error_output:
            logger.warning(f"Error output from code execution: {error_output}")
            result += f"\nError output:\n{error_output}"

        logger.info("Docker code execution completed")
        return result
    
    except Exception as e:
        logger.error(f"Unexpected error in Docker execution: {e}")
        return f"Unexpected error: {e}"

def cleanup_docker_resources():
    if not config.getboolean('Docker', 'EnableCleanup', fallback=False):
        logger.info("Docker cleanup is disabled. Skipping.")
        return

    keep_last_n = config.getint('Docker', 'KeepLastNImages', fallback=5)
    try:
        client = docker.from_env()

        # Remove old containers
        containers = client.containers.list(all=True)
        for container in containers:
            if container.status == 'exited':
                container.remove()
                logger.info(f"Removed Docker container: {container.id}")

        # Remove dangling images
        images = client.images.list(filters={'dangling': True})
        for image in images:
            client.images.remove(image.id)
            logger.info(f"Removed dangling Docker image: {image.id}")

        # Remove old images, keeping the last n
        images = client.images.list()
        sorted_images = sorted(images, key=lambda x: x.attrs['Created'], reverse=True)
        for image in sorted_images[keep_last_n:]:
            try:
                client.images.remove(image.id)
                logger.info(f"Removed old Docker image: {image.id}")
            except docker.errors.APIError as e:
                logger.warning(f"Could not remove Docker image {image.id}: {e}")

    except Exception as e:
        logger.error(f"Error during Docker resource cleanup: {e}")

def log_docker_resource_usage():
    try:
        client = docker.from_env()
        
        # Log container usage
        containers = client.containers.list(all=True)
        logger.info(f"Total containers: {len(containers)}")
        logger.info(f"Running containers: {len([c for c in containers if c.status == 'running'])}")
        
        # Log image usage
        images = client.images.list()
        logger.info(f"Total images: {len(images)}")
        
        # Log disk usage
        total, used, free = shutil.disk_usage("/")
        logger.info(f"Disk usage: {used // (2**30)}GB used, {free // (2**30)}GB free")
    
    except Exception as e:
        logger.error(f"Error logging Docker resource usage: {e}")

# AI interaction functions
def extract_json(text):
    """Extract JSON object from text."""
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        json_str = text[start:end]
        return json.loads(json_str)
    except ValueError:
        return None

def get_ai_response(prompt, api_key):
    ai_provider = config.get('AI', 'PROVIDER', fallback='claude').lower()
    
    if ai_provider == 'claude':
        return get_claude_response(prompt, api_key)
    elif ai_provider == 'openai':
        return get_openai_response(prompt, api_key)
    else:
        error_message = f"Unsupported AI provider: {ai_provider}"
        logger.error(error_message)
        return f"Error: {error_message}"

def get_claude_response(prompt, api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        model = config.get('Anthropic', 'MODEL', fallback='claude-2.1')
        max_output_tokens = min(
            int(config.get('Anthropic', 'MAX_OUTPUT_TOKENS', fallback='4096')),
            MODEL_MAX_OUTPUT_TOKENS.get(model, 4096)
        )
        temperature = float(config.get('Anthropic', 'TEMPERATURE', fallback='0.7'))
        
        logger.info(f"Sending request to Claude AI using model: {model} with max_output_tokens: {max_output_tokens}")
        
        if model.startswith("claude-3"):
            # Use Messages API for Claude 3.x models
            message = client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_content = message.content[0].text
        else:
            # Use Completions API for Claude 2.x models
            response = client.completions.create(
                model=model,
                max_tokens_to_sample=max_output_tokens,
                temperature=temperature,
                prompt=f"Human: {prompt}\n\nAssistant:",
                stop_sequences=["Human:"]
            )
            response_content = response.completion
        
        logger.info("Claude AI response received")
        return response_content
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {str(e)}")
        return f"Error: Anthropic API error - {str(e)}"
    except Exception as e:
        logger.error(f"Error getting Claude AI response: {str(e)}")
        return f"Error: {str(e)}"

def get_openai_response(prompt, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        model = config.get('OpenAI', 'MODEL', fallback='gpt-3.5-turbo')
        max_tokens = int(config.get('OpenAI', 'MAX_TOKENS', fallback='4096'))
        temperature = float(config.get('OpenAI', 'TEMPERATURE', fallback='0.7'))
        
        logger.info(f"Sending request to OpenAI using model: {model} with max_tokens: {max_tokens}")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        response_content = response.choices[0].message.content
        logger.info("OpenAI response received")
        return response_content
    except openai.OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        return f"Error: OpenAI API error - {str(e)}"
    except Exception as e:
        logger.error(f"Error getting OpenAI response: {str(e)}")
        return f"Error: {str(e)}"

def get_ai_prompt(experiment_id, prev_data, action_history, current_dockerfile, current_action, max_actions, time_remaining, access_info):
    network_access = config.getboolean('Docker', 'NetworkAccess', fallback=False)
    prompt = f"""
    Experiment #{experiment_id}

    You are Ouroboros, an advanced AI system designed to self-improve and push the boundaries of AI research through innovative experiments.

    Current Status:
    - Action: {current_action} of {max_actions}
    - Time remaining: {time_remaining:.2f} seconds
    - Network access: {"Enabled" if network_access else "Disabled"}

    Previous experiment data:
    {json.dumps(prev_data, indent=2) if prev_data else 'No previous experiment'}

    Current experiment action history:
    {json.dumps(action_history, indent=2)}

    Current Dockerfile:
    {current_dockerfile}

    IMPORTANT INSTRUCTIONS:
    1. **JSON Format:** All responses must be in valid JSON format. Do not include anything outside the JSON object.
    2. **Docker Environment:**
       - You are working with a persistent Docker container throughout the experiment.
       - Modify the Dockerfile using the "dockerfile" action. This will create a new container with the updated environment.
       - The main experiment code will be copied as 'experiment.py' automatically.
       - Use ENV in the Dockerfile for necessary environment variables.
    3. **Code Execution:** Use the "run" action to execute Python code in the Docker environment. The code will run in the current container.
    4. **Network Access:** No external URLs or APIs if disabled; "google" and "loadurl" actions depend on operator enablement.
    5. **API Keys and Credentials:** Use them cautiously and only if required for the experiment.
    {json.dumps(access_info, indent=2)}
    6. **Experiment Lifecycle:**
       - Each experiment runs in isolation with a maximum of {max_actions} actions.
       - Use "finalize" on the last action or if you achieve your goals early.

    **EXPERIMENTATION GUIDELINES:**
    - Be creative and ambitious. Start simple if it's your first experiment.
    - Build on previous results, focusing on self-improvement and advancing AI research.
    - Explore areas such as novel ML algorithms, efficient data processing, NLP, reasoning, meta-learning, or even fun experiments.
    - Document your process and findings in your code comments and final notes.

    **RESPONSE FORMAT:**
    Respond with a JSON object containing one action with "action," "data," and "notes" fields. Possible actions:
    1. "dockerfile": Update the Dockerfile (creates a new container with the updated environment)
    2. "run": Execute Python code (runs in the current container)
    3. "search": Search previous experiments
    4. "google": Perform a Google search
    5. "loadurl": Load a webpage
    6. "finalize": Conclude the experiment

    Example response format:
    {{
        "action": "dockerfile",
        "data": "FROM python:3.9-slim\\nWORKDIR /app\\nRUN pip install numpy",
        "notes": "Adding numpy for numerical computations."
    }}

    What would you like to do next? Be bold, creative, and aim for breakthroughs!
    """
    return prompt

# Action functions
def search_previous_experiments(query):
    conn = None
    try:
        conn = sqlite3.connect('ouroboros.db')
        c = conn.cursor()
        c.execute("""
            SELECT id, substr(code, 1, 200) as code_preview, substr(results, 1, 200) as results_preview, 
                   ai_notes, substr(dockerfile, 1, 200) as dockerfile_preview, timestamp 
            FROM experiments 
            WHERE code LIKE ? OR results LIKE ? OR ai_notes LIKE ? OR dockerfile LIKE ?
            ORDER BY id DESC
            LIMIT ?
        """, ('%' + query + '%', '%' + query + '%', '%' + query + '%', '%' + query + '%', 
              int(config.get('Search', 'MaxResults', fallback='5'))))
        results = c.fetchall()
        logger.info(f"Search completed for query: {query}")
        return [dict(zip(['id', 'code_preview', 'results_preview', 'ai_notes', 'dockerfile_preview', 'timestamp'], row)) for row in results]
    except sqlite3.Error as e:
        logger.error(f"Database search error: {e}")
        return []
    finally:
        if conn:
            conn.close()

# Status update function
def print_status_update(experiment_id, action_count, max_actions, time_remaining):
    status_message = f"\rExperiment {experiment_id}: Action {action_count}/{max_actions} | Time remaining: {time_remaining:.2f}s"
    sys.stdout.write(status_message)
    sys.stdout.flush()
    logger.info(status_message.strip())

# Experiment functions
def setup_experiment(experiment_id, repo_dir):
    exp_dir = os.path.abspath(os.path.join(repo_dir, f'experiment_{experiment_id}'))
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Created experiment directory: {exp_dir}")

    if not exp_dir.startswith(repo_dir):
        raise ValueError(f"Experiment directory {exp_dir} is not within the Git repository {repo_dir}")

    return exp_dir

def get_previous_experiment_data(experiment_id):
    with sqlite3.connect('ouroboros.db') as conn:
        c = conn.cursor()
        c.execute("SELECT code, results, ai_notes, dockerfile FROM experiments WHERE id = ?", (experiment_id - 1,))
        return c.fetchone()

def save_experiment_results(experiment_id, action_history, results, final_notes, dockerfile):
    with sqlite3.connect('ouroboros.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO experiments (id, code, results, ai_notes, dockerfile, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                  (experiment_id, json.dumps(action_history), results, final_notes, dockerfile, datetime.now().isoformat()))
        conn.commit()

def save_experiment_log(exp_dir, experiment_id, action_history, results, final_notes, dockerfile):
    log_path = os.path.join(exp_dir, 'experiment_log.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump({
            "experiment_id": experiment_id,
            "action_history": action_history,
            "results": results,
            "final_notes": final_notes,
            "dockerfile": dockerfile
        }, f, indent=2)
    logger.info(f"Experiment log saved to {log_path}")
    return log_path

# Main experiment cycle
def run_ai_interaction_loop(experiment_id, prev_data, exp_dir, repo, access, docker_client):
    action_history = []
    current_dockerfile = "FROM python:3.9-slim\nWORKDIR /app\n"
    results = ""
    final_notes = ""
    current_image = None
    current_container = None

    max_actions = int(config.get('Experiment', 'MaxActions', fallback='10'))
    time_limit = float(config.get('Experiment', 'TimeLimit', fallback='3600'))
    start_time = time.time()
    error_count = 0
    max_errors = int(config.get('Experiment', 'MaxErrors', fallback='3'))

    def extract_json(text):
        """Extract JSON object from text."""
        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            json_str = text[start:end]
            return json.loads(json_str)
        except ValueError:
            return None

    def handle_dockerfile_action(data):
        nonlocal current_dockerfile, current_image, current_container
        dockerfile_path = os.path.join(exp_dir, 'Dockerfile')
        os.makedirs(os.path.dirname(dockerfile_path), exist_ok=True)
        with open(dockerfile_path, 'w') as f:
            f.write(data)
        if os.path.exists(dockerfile_path):
            commit_to_git(repo, f"Experiment {experiment_id}: Update Dockerfile", [dockerfile_path])
            logger.info(f"Dockerfile updated and committed for experiment {experiment_id}")
            
            # Clean up previous container and image if they exist
            if current_container:
                current_container.remove(force=True)
                current_container = None
            if current_image:
                docker_client.images.remove(current_image.id, force=True)
                current_image = None

            # Build new image
            logger.info("Building new Docker image...")
            try:
                current_image, _ = docker_client.images.build(
                    path=exp_dir,
                    dockerfile='Dockerfile',
                    rm=True
                )
                logger.info(f"New Docker image built successfully: {current_image.id}")

                # Create new container
                logger.info("Creating new Docker container...")
                current_container = docker_client.containers.create(
                    current_image.id,
                    command="tail -f /dev/null",  # Keep container running
                    volumes={os.path.abspath(exp_dir): {'bind': '/app', 'mode': 'rw'}},  # Changed 'ro' to 'rw'
                    mem_limit=config.get('Docker', 'MemoryLimit', fallback='512m'),
                    cpu_period=100000,
                    cpu_quota=int(config.get('Docker', 'CPUQuota', fallback='50000')),
                    network_mode='none'
                )
                logger.info(f"New Docker container created: {current_container.id}")

                # Start the container
                current_container.start()
                logger.info("New Docker container started successfully")

                current_dockerfile = data
                return "Dockerfile updated and new container is ready for code execution."
            except docker.errors.BuildError as e:
                logger.error(f"Docker build error: {str(e)}")
                return f"Error building Docker image: {str(e)}"
            except docker.errors.APIError as e:
                logger.error(f"Docker API error: {str(e)}")
                return f"Error in Docker operation: {str(e)}"
        else:
            logger.error(f"Failed to create Dockerfile at {dockerfile_path}")
            return "Failed to create Dockerfile"

    def handle_run_action(data):
        nonlocal results
        if not current_container:
            return "No Docker container available. Please update the Dockerfile first."

        results = run_in_docker(docker_client, current_container, data, exp_dir)

        results_path = os.path.join(exp_dir, 'results.txt')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            f.write(results)

        commit_to_git(repo, f"Experiment {experiment_id}: Run code", [results_path])
        logger.info(f"Code executed for experiment {experiment_id}")
        logger.info(f"{'='*50}")
        logger.info(f"Docker execution results for experiment {experiment_id}:")
        logger.info(results)
        logger.info(f"{'='*50}")

        return results

    def handle_search_action(data):
        search_results = search_previous_experiments(data)
        results = json.dumps(search_results, indent=2)
        logger.info(f"Search performed for query: {data}")
        return results

    def handle_google_action(data):
        search_results = google_search(data)
        results = json.dumps(search_results, indent=2)
        logger.info(f"Google search performed for query: {data}")
        return results

    def handle_loadurl_action(data):
        webpage_content = load_webpage(data)
        logger.info(f"Webpage loaded: {data}")
        return webpage_content

    def handle_finalize_action(data):
        logger.info(f"Experiment {experiment_id} finalized. Waiting for next cycle...")
        return "FINALIZE"

    action_handlers = {
        'dockerfile': handle_dockerfile_action,
        'run': handle_run_action,
        'search': handle_search_action,
        'google': handle_google_action,
        'loadurl': handle_loadurl_action,
        'finalize': handle_finalize_action
    }

    ai_provider = config.get('AI', 'PROVIDER', fallback='claude').lower()
    api_key = access['ANTHROPIC_API_KEY'] if ai_provider == 'claude' else access['OPENAI_API_KEY']

    # Test initial Dockerfile
    logger.info("Testing initial Dockerfile...")

    initial_container = None
    try:
        # Create initial container
        logger.info("Creating initial Docker container...")
        initial_container = docker_client.containers.create(
            "python:3.9-slim",
            command="tail -f /dev/null",  # Keep container running
            volumes={os.path.abspath(exp_dir): {'bind': '/app', 'mode': 'rw'}},
            mem_limit=config.get('Docker', 'MemoryLimit', fallback='512m'),
            cpu_period=100000,
            cpu_quota=int(config.get('Docker', 'CPUQuota', fallback='50000')),
            network_mode='none'
        )
        logger.info(f"Initial Docker container created: {initial_container.id}")

        # Start the container
        initial_container.start()
        logger.info("Initial Docker container started successfully")

        # Run the test
        test_code = "print('Initial Dockerfile test successful')"
        test_result = run_in_docker(docker_client, initial_container, test_code, exp_dir)
        logger.info(f"Initial Dockerfile test result: {test_result}")

    except docker.errors.APIError as e:
        logger.error(f"Failed to create or start initial Docker container: {str(e)}")
        test_result = f"Error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error during initial Docker container setup: {str(e)}")
        test_result = f"Unexpected error: {str(e)}"
    finally:
        # Clean up initial container
        if initial_container:
            logger.info("Cleaning up initial Docker container...")
            try:
                initial_container.stop()
                initial_container.remove()
                logger.info("Initial Docker container removed")
            except Exception as e:
                logger.error(f"Error cleaning up initial Docker container: {str(e)}")

    # Main AI interaction loop
    for action_count in range(1, max_actions + 1):
        time_elapsed = time.time() - start_time
        time_remaining = max(0, time_limit - time_elapsed)
        time.sleep(1)  # Added delay between actions

        print_status_update(experiment_id, action_count, max_actions, time_remaining)

        if time_remaining <= 0:
            logger.warning(f"Experiment {experiment_id} reached time limit.")
            break

        ai_response = get_ai_response(get_ai_prompt(
            experiment_id, prev_data, action_history, current_dockerfile, 
            action_count, max_actions, time_remaining, access
        ), api_key)
        
        logger.info(f"Full AI response:\n{ai_response}")  # Log the full response
        
        try:
            response_json = extract_json(ai_response)
            if response_json is None:
                raise ValueError("No valid JSON found in the response")

            if not all(key in response_json for key in ['action', 'data', 'notes']):
                raise ValueError("Invalid JSON structure: missing required fields")
            
            action_type = response_json['action']
            action_data = response_json['data']
            action_notes = response_json['notes']
            
            if action_type in action_handlers:
                result = action_handlers[action_type](action_data)
                if result == "FINALIZE":  # Finalize action
                    final_notes = action_notes + "\n" + action_data
                    action_history.append({
                        "action": action_type,
                        "data": action_data,
                        "notes": action_notes,
                        "results": "Experiment finalized"
                    })
                    break
                elif result:
                    results = result
                action_history.append({
                    "action": action_type,
                    "data": action_data,
                    "notes": action_notes,
                    "results": results
                })
            else:
                results = f"Unknown action in AI response: {action_type}"
                logger.warning(results)
                action_history.append({
                    "action": "UNKNOWN",
                    "data": action_data,
                    "notes": action_notes,
                    "results": results
                })
        
        except json.JSONDecodeError as json_error:
            logger.error(f"Error decoding JSON response: {str(json_error)}")
            error_count += 1
        except ValueError as value_error:
            logger.error(f"Error processing AI response: {str(value_error)}")
            error_count += 1
        except Exception as action_error:
            logger.error(f"Error processing action {action_count}: {str(action_error)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            action_history.append({"action": "ERROR", "error": str(action_error)})
            error_count += 1

        if error_count >= max_errors:
            logger.error(f"Stopping experiment after {max_errors} consecutive errors")
            break

        error_count = 0  # Reset error count after successful processing of all actions

    # Clean up Docker resources at the end of the experiment
    if current_container:
        current_container.remove(force=True)
    if current_image:
        docker_client.images.remove(current_image.id, force=True)

    # If we've reached this point without finalizing, add a final note
    if not final_notes:
        final_notes = f"Experiment ended after {len(action_history)} actions without explicit finalization."

    return action_history, results, final_notes, current_dockerfile

def run_experiment_cycle(docker_client):
    free_space = check_disk_space()
    min_space_gb = config.getint('Docker', 'MinimumDiskSpaceGB', fallback=10)
    if free_space < min_space_gb:
        logger.error(f"Not enough disk space to run experiment. Available: {free_space}GB, Required: {min_space_gb}GB. Aborting.")
        return

    log_docker_resource_usage()
    cleanup_docker_resources()
    log_docker_resource_usage()

    logger.info("Starting new experiment cycle")
    access = read_access()
    experiment_id = get_last_experiment_id() + 1
    logger.info(f"Experiment ID: {experiment_id}")
    
    ai_provider = config.get('AI', 'PROVIDER', fallback='claude').lower()
    if ai_provider == 'claude':
        temperature = config.get('Anthropic', 'TEMPERATURE', fallback='0.7')
    else:
        temperature = config.get('OpenAI', 'TEMPERATURE', fallback='0.7')
    logger.info(f"Current AI temperature setting: {temperature}")
    
    repo, repo_dir = init_git_repo()
    if repo is None or repo_dir is None:
        logger.error("Failed to initialize Git repository. Exiting experiment cycle.")
        return

    try:
        exp_dir = setup_experiment(experiment_id, repo_dir)
        
        if not os.path.exists(exp_dir):
            logger.error(f"Experiment directory does not exist: {exp_dir}")
            return
        
        prev_data = get_previous_experiment_data(experiment_id)
        if prev_data:
            logger.info(f"Retrieved data from previous experiment (ID: {experiment_id - 1})")
        else:
            logger.info("No previous experiment data available")
        
        action_history, results, final_notes, dockerfile = run_ai_interaction_loop(
            experiment_id, prev_data, exp_dir, repo, access, docker_client)

        save_experiment_results(experiment_id, action_history, results, final_notes, dockerfile)
        log_path = save_experiment_log(exp_dir, experiment_id, action_history, results, final_notes, dockerfile)
        commit_to_git(repo, f"Experiment {experiment_id}: Save experiment log", [log_path])
        
        logger.info(f"Experiment {experiment_id} completed with {len(action_history)} actions")
        logger.info(f"Final notes: {final_notes}")
        
    except Exception as e:
        logger.error(f"Error in experiment cycle: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


# Main function
def main():
    if not check_ai_libraries():
        return
    
    init_db()
    docker_client = create_docker_client()
    if not docker_client:
        logger.error("Failed to create Docker client. Exiting.")
        return

    logger.info("Ouroboros system started")
    print("Ouroboros system started. Press Ctrl+C to exit.")

    interval_minutes = int(config.get('Scheduling', 'IntervalMinutes', fallback='60'))
    run_first_immediately = config.getboolean('Scheduling', 'RunFirstImmediately', fallback=False)

    logger.info(f"Experiment interval set to {interval_minutes} minutes")
    print(f"Experiments will run every {interval_minutes} minutes")

    if run_first_immediately:
        logger.info("First experiment will run immediately")
        print("First experiment will run immediately")
    else:
        logger.info("First experiment will run after the initial interval")
        print("First experiment will run after the initial interval")

    try:
        with tqdm(total=0, unit="cycles", desc="Running experiments") as pbar:
            last_run = time.time() - interval_minutes * 60 if run_first_immediately else time.time()
            while True:
                now = time.time()
                if now - last_run >= interval_minutes * 60:
                    logger.info("Starting new experiment cycle")
                    print("\nStarting new experiment cycle...")
                    run_experiment_cycle(docker_client)
                    last_run = now
                    pbar.update(1)
                else:
                    remaining = int(interval_minutes * 60 - (now - last_run))
                    print(f"\rTime until next experiment: {remaining // 60:02d}:{remaining % 60:02d}", end="", flush=True)
                schedule.run_pending()
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Ouroboros system stopped by user.")
        print("\nOuroboros system stopped. Goodbye!")

if __name__ == "__main__":
    main()