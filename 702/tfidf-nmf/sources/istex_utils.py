#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for extracting and processing ISTEX data.
"""

import os
import json
from typing import Dict, List, Tuple
from .europresse_utils import (
    remove_urls_hashtags_emojis_mentions_emails,
    transform_text,
)


def get_nested(data: dict, keys: list):
    """
    Utility function to extract nested data from a dictionary.

    Args:
        data: source dictionary
        keys: list of keys for nested navigation (e.g. ['host', 'title'])

    Returns:
        Extracted value or None if not found.
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return None
    return data


def extract_istex_articles(
    directory_path: str,
    min_chars: int = 500,
    max_chars: int = 200000,
) -> Tuple[List[str], Dict[str, List]]:
    """
    Dedicated extraction for ISTEX data.

    For each subdirectory, search for pairs of .cleaned (content) and .json (metadata) files.

    Args:
        directory_path: path to the folder containing ISTEX subdirectories
        min_chars: minimum number of characters for a document
        max_chars: maximum number of characters for a document
    
    Returns:
        - documents: list of cleaned texts
        - columns_dict: dictionary containing all metadata per field
    """
    print(f"\nISTEX extraction from {directory_path}...")
    
    # ISTEX fields to extract
    fields_to_extract = [
        'date', 'title', 'doi', 'journal', 'language', 'originalGenre',
        'accessCondition', 'pdfVersion', 'abstractCharCount', 'pdfPageCount',
        'pdfWordCount', 'score', 'pdfText', 'imageCount', 'refCount',
        'sectionCount', 'paragraphCount', 'tableCount', 'categories_scopus',
        'categories_scienceMetrix', 'host_volume', 'host_issue',
        'host_publisher', 'host_pages_first', 'host_pages_last', 'host_title',
        'refBibs_count',
    ]
    
    field_mappings = {
        'date': ['publicationDate'],
        'title': ['title'],
        'doi': ['doi'],
        'journal': ['host', 'title'],
        'language': ['language'],
        'originalGenre': ['originalGenre'],
        'accessCondition': ['accessCondition', 'value'],
        'pdfVersion': ['qualityIndicators', 'pdfVersion'],
        'abstractCharCount': ['qualityIndicators', 'abstractCharCount'],
        'pdfPageCount': ['qualityIndicators', 'pdfPageCount'],
        'pdfWordCount': ['qualityIndicators', 'pdfWordCount'],
        'score': ['qualityIndicators', 'score'],
        'pdfText': ['qualityIndicators', 'pdfText'],
        'imageCount': ['qualityIndicators', 'xmlStats', 'imageCount'],
        'refCount': ['qualityIndicators', 'xmlStats', 'refCount'],
        'sectionCount': ['qualityIndicators', 'xmlStats', 'sectionCount'],
        'paragraphCount': ['qualityIndicators', 'xmlStats', 'paragraphCount'],
        'tableCount': ['qualityIndicators', 'xmlStats', 'tableCount'],
        'categories_scopus': ['categories', 'scopus'],
        'categories_scienceMetrix': ['categories', 'scienceMetrix'],
        'host_volume': ['host', 'volume'],
        'host_issue': ['host', 'issue'],
        'host_publisher': ['host', 'publisher'],
        'host_pages_first': ['host', 'pages', 'first'],
        'host_pages_last': ['host', 'pages', 'last'],
        'host_title': ['host', 'title'],
        'refBibs_count': ['refBibs'],
    }
    
    documents = []
    columns_dict = {}
    
    # Initialize columns_dict with empty lists
    for field in fields_to_extract:
        columns_dict[field] = []
    
    if not os.path.isdir(directory_path):
        print(f"❌ Path '{directory_path}' is not a valid directory.")
        return documents, columns_dict
    
    # List all subdirectories
    subdirs = [
        d for d in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, d))
    ]
    
    if not subdirs:
        print(f"❌ No subdirectories found in {directory_path}")
        return documents, columns_dict
    
    print(f"📁 {len(subdirs)} subdirectory(ies) found")
    
    nb_processed = 0
    nb_skipped = 0
    nb_too_short = 0
    nb_too_long = 0
    
    for subdir in subdirs:
        subdir_path = os.path.join(directory_path, subdir)
        
        # List .cleaned and .json files
        files_in_dir = os.listdir(subdir_path)
        txt_files = [f for f in files_in_dir if f.endswith('.cleaned')]
        json_files = [f for f in files_in_dir if f.endswith('.json')]
        
        # Look for common basenames
        txt_basenames = set(os.path.splitext(f)[0] for f in txt_files)
        json_basenames = set(os.path.splitext(f)[0] for f in json_files)
        common_basenames = txt_basenames.intersection(json_basenames)
        
        if not common_basenames:
            nb_skipped += 1
            continue
        
        for basename in common_basenames:
            txt_file_path = os.path.join(subdir_path, basename + '.cleaned')
            json_file_path = os.path.join(subdir_path, basename + '.json')
            
            # Read text file
            try:
                with open(txt_file_path, 'r', encoding='utf-8') as f:
                    txt_content = f.read()
            except Exception as e:
                print(f"⚠️  Error reading '{txt_file_path}': {e}")
                continue
            
            # Length filtering
            text_len = len(txt_content or "")
            if text_len < min_chars:
                nb_too_short += 1
                continue
            if text_len > max_chars:
                nb_too_long += 1
                continue
            
            # Text cleaning
            txt_content = remove_urls_hashtags_emojis_mentions_emails(txt_content)
            txt_content = transform_text(txt_content)
            documents.append(txt_content)
            
            # Read JSON
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception as e:
                print(f"⚠️  Error reading '{json_file_path}': {e}")
                # Fill with None for every field
                for field in fields_to_extract:
                    columns_dict[field].append(None)
                continue
            
            # Field extraction
            for field in fields_to_extract:
                json_keys = field_mappings.get(field)
                value = None
                if json_keys is not None:
                    if field == 'refBibs_count':
                        # Number of bibliographic references
                        refbibs = get_nested(json_data, json_keys)
                        value = len(refbibs) if refbibs is not None else 0
                    else:
                        value = get_nested(json_data, json_keys)
                        # If it's a list, join with commas
                        if isinstance(value, list):
                            value = ', '.join(map(str, value))
                columns_dict[field].append(value)
            
            nb_processed += 1
    
    # Check lengths
    length_documents = len(documents)
    for field in columns_dict:
        if len(columns_dict[field]) != length_documents:
            print(
                f"⚠️  Inconsistency for field '{field}': "
                f"{len(columns_dict[field])} vs {length_documents}"
            )
    
    # Format dates so we keep only the year (YYYY)
    if 'date' in columns_dict:
        formatted_dates = []
        for date in columns_dict['date']:
            if date:
                # Extract just the year if format is YYYY or YYYY-MM-DD
                year = str(date)[:4] if len(str(date)) >= 4 else str(date)
                formatted_dates.append(year)
            else:
                formatted_dates.append(None)
        columns_dict['date'] = formatted_dates
    
    print(
        f"✓ {nb_processed} ISTEX documents extracted, "
        f"{nb_skipped} subdirectories skipped (no .cleaned/.json pairs)"
    )
    if nb_too_short or nb_too_long:
        print(
            f"  ({nb_too_short} documents ignored because < {min_chars} characters, "
            f"{nb_too_long} documents ignored because > {max_chars} characters)"
        )
    
    return documents, columns_dict


def format_istex_metadata(columns_dict: Dict[str, List], index: int, raw_text: str) -> Dict[str, str]:
    """
    Format ISTEX metadata for a document in a shape compatible with Europresse.

    Args:
        columns_dict: metadata dictionary
        index: document index
        raw_text: raw document text

    Returns:
        Dictionary of formatted metadata
    """
    metadata = {
        "Title": str(columns_dict.get('title', ['N/A'])[index]) if index < len(columns_dict.get('title', [])) else "N/A",
        "Authors": "None",  # ISTEX n'a pas d'auteurs dans le même format
        "Raw_authors": "N/A",
        "Journal_original": str(columns_dict.get('journal', ['N/A'])[index]) if index < len(columns_dict.get('journal', [])) else "N/A",
        "Journal_normalized": str(columns_dict.get('journal', ['N/A'])[index]).lower() if index < len(columns_dict.get('journal', [])) else "n/a",
        "Date_normalized": str(columns_dict.get('date', ['N/A'])[index]) if index < len(columns_dict.get('date', [])) else "N/A",
        "Num_characters": str(len(raw_text or "")),
    }
    
    # Remove semicolons
    for key in metadata:
        if isinstance(metadata[key], str):
            metadata[key] = metadata[key].replace(";", "")
    
    return metadata

