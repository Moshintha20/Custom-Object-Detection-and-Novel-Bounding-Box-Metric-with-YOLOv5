import os
import subprocess
import argparse

def run_command(command):
    """Run a shell command and print output in real time."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv5 on a custom dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to the data.yaml file")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--weights", type=str, required=True, help="Path to pretrained weights")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    args = parser.parse_args()

    # Clone the modified YOLOv5 repository if not present
    if not os.path.exists("yolov5"):
        print("Cloning customized YOLOv5 repository...")
        run_command("git clone https://github.com/YOUR_USERNAME/YOUR_YOLOV5_REPO.git yolov5")

    # Change directory to YOLOv5
    os.chdir("yolov5")

    # Run training
    print("Starting YOLOv5 training...")
    train_command = (
        f"python train.py --data {args.data} --epochs {args.epochs} "
        f"--weights {args.weights} --cfg yolov5s.yaml --batch-size {args.batch_size}"
    )
    run_command(train_command)

if __name__ == "__main__":
    main()
