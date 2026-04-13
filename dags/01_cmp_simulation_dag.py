"""
DAG 01: CMP Data Simulator
Generates continuous CMP sensor data with a state machine
that transitions between normal, degrading, and clogging modes.
"""
import os
import json
import random
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


BASE_DIR = "/opt/airflow/data"
OUTPUT_DIR = os.path.join(BASE_DIR, "raw_logs")
STATE_FILE = os.path.join(BASE_DIR, "simulation_state.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_or_init_state():
    """Load previous state or initialize a healthy machine state."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "health": 100.0,
        "flow_rate": 200.0,
        "motor_current": 15.0,
        "mode": "normal",
        "step_count": 0
    }


def save_state(state):
    """Persist state for the next task run."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


def generate_continuous_data(**context):
    """Generate a batch of 60 continuous CMP sensor readings."""
    state = load_or_init_state()
    data_batch = []

    for i in range(60):
        current_time = (datetime.now() + timedelta(seconds=i)).isoformat()

        state["step_count"] += 1

        # Mode transition every ~300 steps
        if state["step_count"] % 300 == 0:
            dice = random.random()
            if dice < 0.7:
                state["mode"] = "normal"
            elif dice < 0.9:
                state["mode"] = "degrading"
            else:
                state["mode"] = "clogging"

        # Simulate sensor readings based on mode
        if state["mode"] == "normal":
            target_flow = 200.0
            state["flow_rate"] += random.uniform(-1.0, 1.0)
            state["flow_rate"] += (target_flow - state["flow_rate"]) * 0.1
            state["motor_current"] = 15.0 + random.uniform(-0.5, 0.5)

        elif state["mode"] == "degrading":
            state["flow_rate"] -= random.uniform(0.05, 0.2)
            state["motor_current"] += random.uniform(0.01, 0.05)

        elif state["mode"] == "clogging":
            state["flow_rate"] -= random.uniform(0.5, 2.0)
            state["motor_current"] += random.uniform(0.1, 0.3)

        state["flow_rate"] = max(0.0, state["flow_rate"])

        error_code = None
        status = "RUNNING"

        if state["flow_rate"] < 180:
            error_code = "ALM-3050"
            status = "WARNING"
        if state["flow_rate"] < 100:
            status = "DOWN"

        record = {
            "timestamp": current_time,
            "machine_id": "CMP-TOOL-08",
            "recipe_id": "N3_Oxide_01",
            "metrics": {
                "slurry_flow_rate_ml_min": round(state["flow_rate"], 2),
                "motor_current_amps": round(state["motor_current"], 2),
                "head_down_force_psi": 5.0,
                "platen_temp_c": round(55.0 + random.uniform(-1, 1), 2)
            },
            "status": status,
            "error_code": error_code
        }

        data_batch.append(record)

    save_state(state)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cmp_machine_log_{timestamp_str}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_batch, f, indent=2)

    print(f"Generated data: {filename} (Mode: {state['mode']})")


default_args = {
    'owner': 'data_engineer',
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    '01_cmp_data_simulator',
    default_args=default_args,
    schedule_interval=timedelta(minutes=1),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['CMP', 'Simulator'],
) as dag:

    generate_task = PythonOperator(
        task_id='generate_cmp_logs',
        python_callable=generate_continuous_data,
    )

    trigger_etl_task = TriggerDagRunOperator(
        task_id='trigger_etl_pipeline',
        trigger_dag_id='02_cmp_etl_polars_v2',
        wait_for_completion=False,
    )

    generate_task >> trigger_etl_task
