"""
Copyright 2025 Katrin Renz, Chen Long, Elahe Arani, Oleg Sinavski
Copyright 2025 Shuncheng Tang
Copyright 2025-2026 Clément Darne

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
import time
import ujson
import shutil
from tqdm.autonotebook import tqdm
import re
import datetime
import signal
import sys
import dotenv


# Load env variables from .env file
dotenv.load_dotenv()
REPO_ROOT = os.getenv('REPO_ROOT')
CARLA_ROOT = os.getenv('CARLA_ROOT')
USERNAME = os.getenv('USERNAME')


def kill_all_carla_and_leaderboard():
    # find the CarlaUE4.sh process
    try:
        carla_pids = subprocess.check_output(
            "ps aux | grep CarlaUE4 | grep -v grep | awk '{print $2}'",
            shell=True
        ).decode().split()
        for pid in carla_pids:
            print(f"Killing CARLA process {pid}")
            os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        pass

    # find the leaderboard_evaluator.py process
    try:
        leaderboard_pids = subprocess.check_output(
            "ps aux | grep leaderboard_evaluator.py | grep -v grep | awk '{print $2}'",
            shell=True
        ).decode().split()
        for pid in leaderboard_pids:
            print(f"Killing leaderboard process {pid}")
            os.kill(int(pid), signal.SIGKILL)
    except subprocess.CalledProcessError:
        pass

def signal_handler(sig, frame):
    print("\nCaught Ctrl+C! Killing all CARLA and leaderboard processes...")
    kill_all_carla_and_leaderboard()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

def bash_file_bench2drive(job, port, tm_port):
    cfg = job["cfg"]
    route = job["route"]
    route_id = job["route_id"]
    seed = job["seed"]
    viz_path = job["viz_path"]
    result_file = job["result_file"]
    log_file = job["log_file"]
    err_file = job["err_file"]
    job_file = job["job_file"]
    gpu = 0

    with open(job_file, 'w', encoding='utf-8') as rsh:
        rsh.write(f"""#!/bin/bash
source ~/.bashrc
. /home/{USERNAME}/anaconda3/etc/profile.d/conda.sh
conda activate simlingo


cd {cfg["repo_root"]}


export CARLA_ROOT={cfg["carla_root"]}
export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:{cfg["carla_root"]}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:{cfg["repo_root"]}/Bench2Drive/leaderboard
export PYTHONPATH=$PYTHONPATH:{cfg["repo_root"]}/Bench2Drive/scenario_runner
export SCENARIO_RUNNER_ROOT={cfg["repo_root"]}/Bench2Drive/scenario_runner


export SAVE_PATH={viz_path}


python -u {cfg["repo_root"]}/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator.py --routes={route} \\
--repetitions=1 \\
--track=SENSORS \\
--checkpoint={result_file} \\
--timeout=600 \\
--agent={cfg["agent_file"]} \\
--agent-config={cfg["checkpoint"]} \\
--traffic-manager-seed={seed} \\
--port={port} \\
--gpu-rank={gpu} \\
--traffic-manager-port={tm_port} \\
> {log_file} 2> {err_file}
""")


configs = [
    {
        "agent": "simlingo",
        "checkpoint": f"{REPO_ROOT}/outputs/simlingo/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt",
        "benchmark": "bench2drive",
        "route_path": f"{REPO_ROOT}/leaderboard/data/bench2drive_split",
        "seeds": [1], # TODO: change depending on how many eval seeds you wanna run (paper uses one eval seed on three train seeds)
        "tries": 2,
        "out_root": f"{REPO_ROOT}/eval_results/Bench2Drive",
        "carla_root": f"{CARLA_ROOT}",
        "repo_root": f"{REPO_ROOT}",
        "agent_file": f"{REPO_ROOT}/team_code/agent_simlingo.py",
        "team_code": "team_code",
        "agent_config": "not_used",
        "username": f"{USERNAME}",
    }
]


job_queue = []
for cfg in configs:
    route_path = cfg["route_path"]
    routes = sorted([x for x in os.listdir(route_path) if x.endswith(".xml")],
                    key=lambda x: int(x.split('_')[-1].split('.xml')[0]))


    for seed in cfg["seeds"]:
        base_dir = os.path.join(cfg["out_root"], cfg["agent"], cfg["benchmark"], str(seed))
        os.makedirs(os.path.join(base_dir, "run"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "res"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "out"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "err"), exist_ok=True)


        for route in routes[:1]:
            route_id = route.split("_")[-1][:-4].zfill(3)
            route = os.path.join(route_path, route)


            viz_path = os.path.join(base_dir, "viz", route_id)
            os.makedirs(viz_path, exist_ok=True)

            log_time = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
            result_file = os.path.join(base_dir, "res", f"route_{route_id}_res_{log_time}.json")
            log_file = os.path.join(base_dir, "out", f"route_{route_id}_out_{log_time}.log")
            err_file = os.path.join(base_dir, "err", f"route_{route_id}_err_{log_time}.log")
            job_file = os.path.join(base_dir, "run", f"eval_{route_id}.sh")


            job = {
                "cfg": cfg,
                "route": route,
                "route_id": route_id,
                "seed": seed,
                "viz_path": viz_path,
                "result_file": result_file,
                "log_file": log_file,
                "err_file": err_file,
                "job_file": job_file,
                "tries": cfg["tries"],
            }
            job_queue.append(job)




progress = tqdm(total=len(job_queue))
for job in job_queue:
    bash_file_bench2drive(job, port=2000, tm_port=2500)
    subprocess.run(["bash", job["job_file"]])
    progress.update(1)
    time.sleep(3)
