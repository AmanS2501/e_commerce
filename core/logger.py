import os
import json
from config.config import Config

def log_interaction(turn_data):
    # Logs all turns to a generic session.json file (not user-specific)
    log_dir = Config.LOG_FOLDER
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "session.json")
    if os.path.exists(log_file):
        with open(log_file, "r") as fp:
            logs = json.load(fp)
    else:
        logs = []
    logs.append(turn_data)
    with open(log_file, "w") as fp:
        json.dump(logs, fp, indent=4)

def get_session_history():
    log_dir = Config.LOG_FOLDER
    log_file = os.path.join(log_dir, "session.json")
    if os.path.exists(log_file):
        with open(log_file, "r") as fp:
            return json.load(fp)
    else:
        return []
