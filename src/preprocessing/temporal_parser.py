"""
TempSound Preprocessing Framework - Temporal Caption Parser (Standard Library Edition)
Author: Onyekachi Onyenokwe (TempSound Capstone Project)

This module parses raw AudioCaps metadata CSVs and filters for explicit 
temporal conjunctions and sequence markers using NLTK POS tagging and Regex.
Uses standard library 'csv' module to bypass Windows Application Control DLL blocks.
"""

import os
import re
import csv
import argparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Download NLTK resources safely
def download_nltk_resources():
    required_resources = ['punkt', 'averaged_perceptron_tagger', 'punkt_tab', 'averaged_perceptron_tagger_eng']
    for resource in required_resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception as e:
            print(f"[Warning] Failed to download NLTK resource '{resource}': {e}")

download_nltk_resources()


class TemporalCaptionParser:
    """
    Extracts temporally-explicit captions from raw AudioCaps dataset.
    """
    
    # Primary temporal sequence regex pattern
    TEMPORAL_REGEX = re.compile(
        r'\b(then|after|before|while|during|followed by|prior to|subsequently|meanwhile|next|as soon as)\b',
        flags=re.IGNORECASE
    )

    def __init__(self, input_csv: str, output_csv: str):
        self.input_csv = input_csv
        self.output_csv = output_csv

    def is_temporally_explicit(self, caption: str) -> bool:
        """
        Validates if a caption contains explicit temporal markers and clause structures.
        """
        if not isinstance(caption, str) or not caption.strip():
            return False

        # Step 1: Fast Regex check for primary sequence markers
        if not self.TEMPORAL_REGEX.search(caption):
            return False

        # Step 2: Linguistic POS Validation via NLTK
        try:
            tokens = word_tokenize(caption)
            pos_tags = pos_tag(tokens)

            has_temporal_marker = any(
                word.lower() in {'then', 'after', 'before', 'while', 'followed', 'during'}
                for word, tag in pos_tags
            )
            has_action_verbs = any(
                tag in {'VBG', 'VBD', 'VBN', 'VBZ', 'VBP'} 
                for word, tag in pos_tags
            )

            return has_temporal_marker and has_action_verbs
        except Exception:
            # Fallback to regex if tokenizer fails
            return True

    def extract_temporal_subset(self):
        """
        Reads input CSV line-by-line using Python native 'csv' module,
        filters for temporal captions, and saves the output.
        """
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"Input dataset not found at: {self.input_csv}")

        print(f"[*] Reading dataset from: {self.input_csv}")

        total_rows = 0
        matched_rows = 0

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        with open(self.input_csv, mode='r', encoding='utf-8', errors='replace') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames

            if not fieldnames:
                raise ValueError(f"CSV file at {self.input_csv} is empty or invalid.")

            # Identify caption column ('caption' or 'text')
            caption_col = 'caption' if 'caption' in fieldnames else ('text' if 'text' in fieldnames else None)
            if not caption_col:
                raise KeyError(f"Could not find 'caption' or 'text' column in header: {fieldnames}")

            with open(self.output_csv, mode='w', encoding='utf-8', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in reader:
                    total_rows += 1
                    caption = row.get(caption_col, '')
                    if self.is_temporally_explicit(caption):
                        writer.writerow(row)
                        matched_rows += 1

        print(f"[✓] Parsing complete! Extracted {matched_rows} / {total_rows} temporal captions.")
        print(f"[✓] Filtered dataset saved to: {self.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TempSound - Stage 1 Temporal Caption Parser")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/raw/test.csv", 
        help="Path to raw AudioCaps metadata CSV"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/processed/test_temporal_subset.csv", 
        help="Path to save filtered temporal subset CSV"
    )

    args = parser.parse_args()

    parser_instance = TemporalCaptionParser(input_csv=args.input, output_csv=args.output)
    parser_instance.extract_temporal_subset()