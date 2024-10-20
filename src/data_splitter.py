import splitfolders
import os

input_path = '../data'
output_path = '../split_data'

splitfolders.ratio(input_path, output=output_path, seed=25, ratio=(0.80, 0.20))