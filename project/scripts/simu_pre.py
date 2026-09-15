import pandas as pd
from pathlib import Path
from src import Field

DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)

TOTAL_STEPS = 100000
RECORD_INTERVAL = 1

def run_simulation():
    field = Field()
    logs = []

    print("simulation start...")
    for step_num in range(1, TOTAL_STEPS + 1):
        field.step()
        
        if step_num % RECORD_INTERVAL == 0:
            logs.append({
                "step": step_num,
                "altruism_count": field.altruism_count,
                "population": len(field.agents)
            })
            field.altruism_count = 0

    # logs saved to CSV
    df = pd.DataFrame(logs)
    csv_path = DATA_DIR / "simulation_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"simulation completed. Logs saved to: {csv_path}")

if __name__ == "__main__":
    run_simulation()