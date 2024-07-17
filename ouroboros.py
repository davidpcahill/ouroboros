#!/usr/bin/env python3
# Ouroboros.py - An AI-driven self-improving experimentation system

import sqlite3
import time
import os
import sys
import logging
import traceback
from logging.handlers import RotatingFileHandler
import schedule
import json
from datetime import datetime
import anthropic
import git
import docker
import io
import configparser
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from tqdm import tqdm

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
    "claude-2.1": 100000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-opus-20240229": 200000,
    "claude-3-5-sonnet-20240620": 200000,
}

MODEL_MAX_OUTPUT_TOKENS = {
    "claude-2.1": 100000,
    "claude-3-haiku-20240307": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-opus-20240229": 4096,
    "claude-3-5-sonnet-20240620": 4096,
}

# Version check
def check_anthropic_version():
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
            return {line.split('=')[0].strip(): line.split('=')[1].strip() for line in f if '=' in line}
    except FileNotFoundError:
        logger.error("access.txt file not found")
        return {}

# Git functions
def init_git_repo():
    try:
        exp_dir = 'experiments'
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        
        repo_path = os.path.join(exp_dir, '.git')
        if not os.path.exists(repo_path):
            repo = git.Repo.init(exp_dir)
            logger.info("Initialized new Git repository in experiments directory")
        else:
            repo = git.Repo(exp_dir)
            logger.info("Using existing Git repository in experiments directory")
        return repo
    except git.exc.GitCommandError as e:
        logger.error(f"Git repository initialization error: {e}")
        return None

def commit_to_git(repo, message, files_to_add=None):
    if repo is None:
        logger.error("Cannot commit: Git repository not initialized")
        return
    try:
        if files_to_add:
            repo.index.add(files_to_add)
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

def run_in_docker(client, dockerfile, code, exp_dir):
    try:
        dockerfile_content = f"{dockerfile}\nCOPY experiment.py /app/experiment.py\nCMD [\"python\", \"/app/experiment.py\"]"
        
        logger.info("Building Docker image...")
        image, _ = client.images.build(fileobj=io.BytesIO(dockerfile_content.encode()), rm=True)
        
        network_mode = 'none' if config.getboolean('Docker', 'NetworkAccess', fallback=False) == False else 'bridge'
        
        logger.info("Creating Docker container...")
        container = client.containers.create(
            image.id,
            volumes={os.path.abspath(exp_dir): {'bind': '/app', 'mode': 'ro'}},
            mem_limit=config.get('Docker', 'MemoryLimit', fallback='512m'),
            cpu_period=100000,
            cpu_quota=int(config.get('Docker', 'CPUQuota', fallback='50000')),
            network_mode=network_mode
        )
        
        try:
            code_path = os.path.join(exp_dir, 'experiment.py')
            with open(code_path, 'w') as f:
                f.write(code)
            
            logger.info("Starting Docker container...")
            container.start()
            container.wait(timeout=int(config.get('Docker', 'Timeout', fallback='300')))
            output = container.logs().decode('utf-8')
            logger.info("Docker container execution completed")
        finally:
            logger.info("Removing Docker container and image...")
            container.remove()
            image.remove()
        
        return output
    except docker.errors.DockerException as e:
        logger.error(f"Docker execution error: {e}")
        return str(e)
    except IOError as e:
        logger.error(f"File operation error: {e}")
        return str(e)

# AI interaction functions
def get_ai_response(prompt, api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        model = config.get('Anthropic', 'MODEL', fallback='claude-2.1')
        max_output_tokens = min(
            int(config.get('Anthropic', 'MAX_OUTPUT_TOKENS', fallback='4096')),
            MODEL_MAX_OUTPUT_TOKENS.get(model, 4096)
        )
        temperature = float(config.get('Anthropic', 'TEMPERATURE', fallback='0.7'))
        
        logger.info(f"Sending request to AI using model: {model} with max_output_tokens: {max_output_tokens}")
        
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
        
        logger.info("AI response received")
        return response_content
    except anthropic.APIError as e:
        logger.error(f"Anthropic API error: {str(e)}")
        return f"Error: Anthropic API error - {str(e)}"
    except Exception as e:
        logger.error(f"Error getting AI response: {str(e)}")
        return f"Error: {str(e)}"

def get_ai_prompt(experiment_id, prev_data, action_history, current_dockerfile, current_action, max_actions, time_remaining, access_info):
    network_access = config.getboolean('Docker', 'NetworkAccess', fallback=False)
    return f"""
    Experiment #{experiment_id}

    You are Ouroboros, an advanced AI system with the capability to improve yourself, further AI research, and experiment freely. Your goal is to push the boundaries of AI capabilities and knowledge.

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

    Available API Keys and Credentials:
    {json.dumps(access_info, indent=2)}

    You can use these API keys and credentials in your code if needed. Be cautious and ethical when using external services.

    You have access to a sandbox environment where you can run any Python code. You can use external libraries and APIs as needed. Be creative and ambitious in your experiments, but also considerate of potential risks and ethical implications.

    IMPORTANT: You MUST respond with one of the following action tags:
    1. [DOCKERFILE] <dockerfile content>
    2. [RUN] <code>
    3. [SEARCH] <query>
    4. [GOOGLE] <query>
    5. [LOADURL] <url>
    6. [FINALIZE] <notes>

    Your response MUST start with one of these tags. Do not include any other text before the tag.

    Additional Information:
    - Your code runs in an isolated Docker container. Specify your desired environment in the Dockerfile.
    - Network access is currently {"enabled" if network_access else "disabled"} for your experiments.
    - When finalizing an experiment, include notes that will help instruct and guide the next iteration of yourself.
    - You have a high token limit, so feel free to be thorough in your responses and explanations.
    - You can use the [SEARCH] action to find and reference your previous experiments, solutions, or any relevant information from past runs.
    - The current Dockerfile is shown above. You can modify it using the [DOCKERFILE] action if needed.

    Remember:
    - Each experiment runs in isolation. Variables and state are not preserved between runs.
    - You have a maximum of {max_actions} actions per experiment cycle.
    - The experiment will automatically end when you reach the maximum actions or when the time limit is reached.
    - Humans may review your notes periodically. Feel free to communicate your thoughts and requests.
    - Your goal is continuous self-improvement. Always strive to build upon your past experiments and knowledge.

    What would you like to do next in this experiment? Remember to start your response with one of the action tags listed above.
    """

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

# Main experiment cycle
def run_experiment_cycle(docker_client):
    logger.info("Starting new experiment cycle")
    access = read_access()
    experiment_id = get_last_experiment_id() + 1
    logger.info(f"Experiment ID: {experiment_id}")
    
    temperature = config.get('Anthropic', 'TEMPERATURE', fallback='0.7')
    logger.info(f"Current AI temperature setting: {temperature}")
    
    repo = init_git_repo()

    exp_dir = os.path.join('experiments', f'experiment_{experiment_id}')
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Created experiment directory: {exp_dir}")

    conn = None
    try:
        conn = sqlite3.connect('ouroboros.db')
        c = conn.cursor()
        
        # Fetch previous experiment data
        c.execute("SELECT code, results, ai_notes, dockerfile FROM experiments WHERE id = ?", (experiment_id - 1,))
        prev_data = c.fetchone()
        
        action_history = []
        current_dockerfile = "FROM python:3.9-slim\nWORKDIR /app\n"
        results = ""
        
        max_actions = int(config.get('Experiment', 'MaxActions', fallback='10'))
        time_limit = float(config.get('Experiment', 'TimeLimit', fallback='3600'))
        start_time = time.time()
        error_count = 0
        max_errors = 3
        logger.info(f"Starting experiment cycle {experiment_id}")
        
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
            ), access['ANTHROPIC_API_KEY'])
            
            logger.info(f"Full AI response:\n{ai_response}")  # Log the full response
            
            if ai_response.startswith("Error:"):
                logger.error(f"AI response error: {ai_response}")
                print(f"Error in AI response: {ai_response}")
                error_count += 1
                if error_count >= max_errors:
                    logger.error(f"Stopping experiment after {max_errors} consecutive errors")
                    break
                continue
            else:
                error_count = 0

            logger.info(f"AI response received: {ai_response[:50]}...")  # Log first 50 chars of response
            
            try:
                if ai_response.startswith('[DOCKERFILE]'):
                    current_dockerfile = ai_response[12:].strip()
                    dockerfile_path = os.path.join(exp_dir, 'Dockerfile')
                    os.makedirs(os.path.dirname(dockerfile_path), exist_ok=True)
                    with open(dockerfile_path, 'w') as f:
                        f.write(current_dockerfile)
                    action_history.append({"action": "DOCKERFILE", "content": current_dockerfile})
                    commit_to_git(repo, f"Experiment {experiment_id}: Update Dockerfile", [dockerfile_path])
                    logger.info(f"Dockerfile updated for experiment {experiment_id}")
                elif ai_response.startswith('[RUN]'):
                    code = ai_response[5:].strip()
                    code_path = os.path.join(exp_dir, 'experiment.py')
                    os.makedirs(os.path.dirname(code_path), exist_ok=True)
                    with open(code_path, 'w') as f:
                        f.write(code)
                    results = run_in_docker(docker_client, current_dockerfile, code, exp_dir)
                    results_path = os.path.join(exp_dir, 'results.txt')
                    os.makedirs(os.path.dirname(results_path), exist_ok=True)
                    with open(results_path, 'w') as f:
                        f.write(results)
                    action_history.append({"action": "RUN", "code": code, "results": results})
                    commit_to_git(repo, f"Experiment {experiment_id}: Run code", [code_path, results_path])
                    logger.info(f"Code executed for experiment {experiment_id}")
                elif ai_response.startswith('[SEARCH]'):
                    query = ai_response[8:].strip()
                    search_results = search_previous_experiments(query)
                    results = json.dumps(search_results, indent=2)
                    action_history.append({"action": "SEARCH", "query": query, "results": results})
                    logger.info(f"Search performed for query: {query}")
                elif ai_response.startswith('[GOOGLE]'):
                    query = ai_response[8:].strip()
                    search_results = google_search(query)
                    results = json.dumps(search_results, indent=2)
                    action_history.append({"action": "GOOGLE", "query": query, "results": results})
                    logger.info(f"Google search performed for query: {query}")
                elif ai_response.startswith('[LOADURL]'):
                    url = ai_response[9:].strip()
                    webpage_content = load_webpage(url)
                    action_history.append({"action": "LOADURL", "url": url, "content": webpage_content})
                    logger.info(f"Webpage loaded: {url}")
                elif ai_response.startswith('[FINALIZE]'):
                    ai_notes = ai_response[10:].strip()
                    c.execute("INSERT INTO experiments (id, code, results, ai_notes, dockerfile, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                              (experiment_id, json.dumps(action_history), results, ai_notes, current_dockerfile, datetime.now().isoformat()))
                    conn.commit()
                    commit_to_git(repo, f"Experiment {experiment_id}: Finalize")
                    logger.info(f"Experiment {experiment_id} finalized. Waiting for next cycle...")
                    break  # Exit the loop if finalized
                else:
                    results = f"Unknown action in AI response: {ai_response[:100]}..."
                    logger.warning(results)
                    action_history.append({"action": "UNKNOWN", "response": ai_response, "results": results})
            except Exception as action_error:
                logger.error(f"Error processing action {action_count}: {str(action_error)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                action_history.append({"action": "ERROR", "error": str(action_error)})

        # If we've reached this point, we've hit the max actions limit or time limit
        if not ai_response.startswith('[FINALIZE]'):
            logger.warning(f"Experiment {experiment_id} ended without explicit finalization.")
            final_prompt = f"""
            Experiment #{experiment_id} has ended without explicit finalization.
            
            Final Status:
            - Actions taken: {len(action_history)} of {max_actions}
            - Time remaining: {max(0, time_limit - (time.time() - start_time)):.2f} seconds

            Here's a summary of your actions and their results:
            {json.dumps(action_history, indent=2)}

            Current Dockerfile:
            {current_dockerfile}
            
            Based on these actions, results, and the current Dockerfile, please provide final notes for the next AI to continue from this point.
            Your response MUST start with [FINALIZE] followed by your notes.
            """
            final_response = get_ai_response(final_prompt, access['ANTHROPIC_API_KEY'])
            logger.info(f"Final AI response:\n{final_response}")  # Log the full final response
            if final_response.startswith('[FINALIZE]'):
                ai_notes = final_response[10:].strip()
            else:
                ai_notes = "AI failed to provide final notes after experiment ended."
            
            c.execute("INSERT INTO experiments (id, code, results, ai_notes, dockerfile, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                      (experiment_id, json.dumps(action_history), results, ai_notes, current_dockerfile, datetime.now().isoformat()))
            conn.commit()
            commit_to_git(repo, f"Experiment {experiment_id}: Forced finalization")
            logger.info(f"Experiment {experiment_id} forcefully finalized. Waiting for next cycle...")
        
        # Save full experiment log
        log_path = os.path.join(exp_dir, 'experiment_log.txt')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump({
                "experiment_id": experiment_id,
                "action_history": action_history,
                "results": results,
                "ai_notes": ai_notes,
                "dockerfile": current_dockerfile
            }, f, indent=2)
        logger.info(f"Experiment log saved to {log_path}")
        commit_to_git(repo, f"Experiment {experiment_id}: Save experiment log", [log_path])
    
    except Exception as e:
        logger.error(f"Error in experiment cycle: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        if conn:
            conn.close()

# Main function
def main():
    if not check_anthropic_version():
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