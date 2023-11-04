import argparse
from data.preprocess_dataset import preprocess_dataset
from models.delete_toxic import delete_toxic


parser = argparse.ArgumentParser(description="")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--size", type=int, default=20)

args = parser.parse_args()
data = preprocess_dataset()
detox = delete_toxic(data, threshold=args.threshold, size=args.size)
print(detox)