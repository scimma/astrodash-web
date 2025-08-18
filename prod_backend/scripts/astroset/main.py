"""
main.py
-------
Main orchestration script for the Astrodash-style training set pipeline.
"""

from config import get_config
from training_set.template_parser import parse_template_list, parse_all_templates
from dataset_builder import build_dataset
from splitter import split_train_test
from saver import save_dataset
import os

def main():
    """
    Main function to run the training set creation pipeline.
    Steps:
    1. Load config
    2. Parse template lists and files
    3. Build dataset arrays
    4. Split into train/test
    5. Save datasets
    """
    # 1. Load config
    config = get_config()
    print("Loaded config.")

    # 2. Parse template list and files
    templist_path = os.path.join(os.path.dirname(__file__), 'training_set', 'templist.txt')
    print(f"Reading template list from {templist_path}")
    filelist = parse_template_list(templist_path)
    filelist = [os.path.join(os.path.dirname(templist_path), f) for f in filelist]
    print(f"Found {len(filelist)} valid template files.")
    spectra_list = parse_all_templates(filelist, config)
    print(f"Parsed {len(spectra_list)} spectra from all templates.")

    # 3. Build dataset arrays
    dataset_dict = build_dataset(spectra_list, config)
    print(f"Built dataset: {dataset_dict['images'].shape[0]} samples.")

    # 4. Split into train/test
    train_dict, test_dict = split_train_test(dataset_dict, train_fraction=0.8)
    print(f"Split into {len(train_dict['labels'])} train and {len(test_dict['labels'])} test samples.")

    # 5. Save datasets
    outdir = os.path.join(os.path.dirname(__file__), 'training_set')
    save_dataset(train_dict, outdir, 'train')
    save_dataset(test_dict, outdir, 'test')
    print(f"Saved train and test datasets to {outdir}.")

if __name__ == "__main__":
    main()
