import subprocess
import sys

if __name__ == "__main__":
    print("starting the complete ingestion process...")
    print("1. ingestion...")
    subprocess.run([sys.executable, "src/ingestion.py"], check=True)
    print("2. preprocessing...")
    subprocess.run([sys.executable, "src/preprocessing.py"], check=True)
    print("3. features...")
    subprocess.run([sys.executable, "src/features.py"], check=True)

    if "--retrain" in sys.argv:
        print("4. training...")
        subprocess.run([sys.executable, "src/train.py"], check=True)