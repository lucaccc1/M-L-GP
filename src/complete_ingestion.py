import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "src/ingestion.py"], check=True)
    subprocess.run([sys.executable, "src/preprocessing.py"], check=True)