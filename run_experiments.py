import subprocess
import os
import time
import json
from datetime import datetime

# Paths to your experiment scripts
EXPERIMENTS = [
    "test/rhea_test.py",
    "test/ultra_def_test.py",
    # add more test scripts here once created
]

RESULTS_DIR = "experiment_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiment(script_path, extra_args=None):
    """
    Runs the experiment scripts and captures their output/logs.
    """
    start_time = time.time()
    cmd = ["python", script_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n Running {script_path}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # don’t crash even if the script errors
        )
    except Exception as e:
        return {"script": script_path, "error": str(e)}

    end_time = time.time()

    # Save logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(
        RESULTS_DIR,
        f"{os.path.basename(script_path).replace('.py', '')}_{timestamp}.log"
    )
    with open(log_filename, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n--- STDERR ---\n")
            f.write(result.stderr)

    # Return metadata for summary
    return {
        "script": script_path,
        "returncode": result.returncode,
        "duration_sec": round(end_time - start_time, 2),
        "log_file": log_filename,
        "stdout_excerpt": result.stdout[:400],
        "stderr_excerpt": result.stderr[:400],
    }

def main():
    summary = []
    for script in EXPERIMENTS:
        res = run_experiment(script)
        summary.append(res)

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n All experiments completed.")
    for r in summary:
        print(f"  - {r['script']} → return {r['returncode']} ({r['duration_sec']}s)")

    print(f"\n Summary written to: {summary_file}")

if __name__ == "__main__":
    main()
