from jtop import jtop
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fname', dest='file_name', type=str, help='Add filename')
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG, filename=str(args.file_name), filemode="a+",format="")
with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    while jetson.ok():
        # Read tegra stats
        logging.info(jetson.stats)
        print(jetson.stats)