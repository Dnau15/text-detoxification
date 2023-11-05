import argparse
from data.make_dataset import preprocess_dataset
from models.delete_toxic import delete_toxic
from utils import save_temp

parser = argparse.ArgumentParser(description="")
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--size", type=int, default=20)

args = parser.parse_args()
data = preprocess_dataset()

toxic_list = data.reference.to_list()
detox = delete_toxic(data, threshold=args.threshold, size=args.size)


print(detox)