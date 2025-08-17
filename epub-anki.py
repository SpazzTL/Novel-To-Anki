import csv
import os
import pickle
import re
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from typing import Any, Dict, List, Tuple

import ebooklib
from ebooklib import epub

# Use a more performant JSON library if available
try:
    import orjson
except ImportError:
    import json as orjson

# --- Constants ---
# Pre-compiled regex for possible speed gain
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
SENTENCE_SPLITTER = re.compile(r'(?<=[.?!])\s+')
DICT_SOURCE_SPLITTER = re.compile(r'(<b>.*?<\/b>:)')

# Output Directories
DICTIONARIES_FOLDER = 'Dictionaries'
CACHE_FOLDER = 'cache'
ANKI_FOLDER = 'output/anki_csv'
GRAMMAR_FOLDER = 'output/grammar_analysis'
JSON_FOLDER = 'output/json_analysis'


def parse_definition(definition_text: str) -> str:
    """
    Parses a raw definition string, which may contain multiple dictionary entries
    in various formats (simple lists, complex JSON), and converts it into clean,
    human-readable HTML.

    Args:
        definition_text: The raw definition string from the 'definitions' field.

    Returns:
        A clean HTML-formatted string summarizing the definitions.
    """
    if not isinstance(definition_text, str) or not definition_text.strip():
        return '<div class="definition-content">No definition found.</div>'

    # Split the raw text by dictionary sources (e.g., "<b>krdict_v2:</b>")
    parts = DICT_SOURCE_SPLITTER.split(definition_text)
    if len(parts) <= 1:
        return f'<div class="definition-content">{definition_text}</div>'

    cleaned_definitions = []

    # Process the parts in chunks of [source, content]
    for i in range(1, len(parts), 2):
        source_tag = parts[i]
        content_str = parts[i + 1].strip()
        
        processed_content = ""

        try:
            data = orjson.loads(content_str)
            if isinstance(data, list) and all(isinstance(item, str) for item in data):
                processed_content = '; '.join(data)
        except (orjson.JSONDecodeError, TypeError):
            # If it fails, it's not a simple list, so we move to the next handler.
            pass

        # --- Handler 2: Try to parse as complex structured content ---
        if not processed_content:
            try:
                # Fix common JSON issues like single quotes
                json_part = content_str.replace("'", '"').replace("`", '"')
                data = orjson.loads(json_part)
                
                descriptions = []
                
                # This recursive function will walk the JSON tree to find description nodes
                def find_english_descriptions(node: Any):
                    if isinstance(node, dict):
                        # Target nodes are divs containing an English string description
                        is_description = (
                            node.get('tag') == 'div' and 
                            node.get('lang') == 'en' and 
                            isinstance(node.get('content'), str)
                        )
                        if is_description:
                            descriptions.append(node['content'])
                        
                        # Also check for numbered meanings (e.g., "1. ", "2. ")
                        is_meaning_number = (
                            node.get('tag') == 'span' and
                            node.get('style', {}).get('fontWeight') == 'bold' and
                            isinstance(node.get('content'), str) and
                            re.match(r'^\d+\.\s*$', node.get('content'))
                        )
                        if is_meaning_number:
                             descriptions.append(f"<br><b>{node['content'].strip()}</b>")

                        # Recurse into child nodes
                        if 'content' in node:
                            find_english_descriptions(node['content'])

                    elif isinstance(node, list):
                        for item in node:
                            find_english_descriptions(item)

                find_english_descriptions(data)
                processed_content = ' '.join(descriptions).replace("<br> ", "<br>")

            except (orjson.JSONDecodeError, IndexError, AttributeError, TypeError):
                # If complex parsing fails, use the raw content as a fallback
                processed_content = content_str # It normally fails 

        if processed_content:
            cleaned_definitions.append(f"{source_tag} {processed_content}")

    return '<div class="definition-content">' + '<br>'.join(cleaned_definitions) + '</div>'


def find_source_files() -> List[str]:
    """Scans directories for .epub and .txt files."""
    files = []
    for directory in ['.', 'input', 'novels']:
        if os.path.isdir(directory):
            files.extend(glob(os.path.join(directory, '*.epub')))
            files.extend(glob(os.path.join(directory, '*.txt')))
    return files


def extract_text_from_epub(epub_path: str) -> str:
    """Extracts all text content from an EPUB file, cleaning HTML tags."""
    text_content = []
    try:
        book = epub.read_epub(epub_path)
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                raw_html = item.get_body_content().decode('utf-8', 'ignore')
                text_content.append(clean_html(raw_html))
            except Exception as e:
                print(f"WARNING: Skipping a malformed item in '{os.path.basename(epub_path)}' due to error: {e}")
                continue
        return "".join(text_content)
    except (zipfile.BadZipFile, KeyError) as e:
        print(f"\nWARNING: Skipping corrupt or malformed EPUB file: {os.path.basename(epub_path)} (Error: {e})")
        return ""
    except Exception as e:
        print(f"\nWARNING: An unexpected error occurred with file '{os.path.basename(epub_path)}': {e}")
        return ""


def extract_text_from_txt(txt_path: str) -> str:
    """Extracts text content from a .txt file."""
    with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def clean_html(raw_html: str) -> str:
    """Removes HTML tags and extra whitespace from a string."""
    cleantext = re.sub(CLEANR, '', raw_html)
    return re.sub(r'\s+', ' ', cleantext).strip()


def get_korean_sentences(text: str) -> List[str]:
    """Splits a block of text into sentences."""
    return [s.strip() for s in SENTENCE_SPLITTER.split(text) if s.strip()]


def initialize_worker():
    """Initializes the KoNLPy Okt tagger for each worker process."""
    global okt
    from konlpy.tag import Okt
    okt = Okt()


def analyze_sentence_chunk(sentence_chunk: List[str]) -> Tuple[Dict[str, Dict], List[str]]:
    """Analyzes a list of sentences to extract words and grammar patterns."""
    words = defaultdict(lambda: {'count': 0, 'sentences': []})
    grammar = []
    for sentence in sentence_chunk:
        try:
            # norm=True (normalize), stem=True (stem words to their root)
            pos_tagged = okt.pos(sentence, norm=True, stem=True)
            for word, pos in pos_tagged:
                # Filter for meaningful parts of speech and ignore single-character words
                if pos in ['Noun', 'Verb', 'Adjective', 'Adverb'] and len(word) > 1:
                    words[word]['count'] += 1
                    if sentence not in words[word]['sentences']:
                        words[word]['sentences'].append(sentence)
                if pos not in ['Punctuation', 'Foreign']:
                    grammar.append(f"{word}/{pos}")
        except Exception as e:
            # Log error but continue processing other sentences
            print(f"\nWARNING: KoNLPy failed to process a sentence. Error: {e}")
            pass
    return dict(words), grammar


def analyze_korean_text_parallel(text: str) -> Tuple[Dict[str, Dict], Counter]:
    """
    Analyzes a large Korean text in parallel to extract word frequencies,
    example sentences, and grammar patterns.
    """
    sentences = get_korean_sentences(text)
    words = defaultdict(lambda: {'count': 0, 'sentences': [], 'definitions': {}})
    grammar = Counter()

    # Split sentences into manageable chunks for parallel processing
    chunk_size = 2000
    sentence_chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]

    with ProcessPoolExecutor(initializer=initialize_worker) as executor:
        futures = {executor.submit(analyze_sentence_chunk, chunk) for chunk in sentence_chunks}
        
        for i, future in enumerate(as_completed(futures), 1):
            sys.stdout.write(f"\r  Processing text chunks: {i}/{len(sentence_chunks)}")
            sys.stdout.flush()

            chunk_words, chunk_grammar = future.result()
            # Merge results from the chunk into the main collection
            for word, data in chunk_words.items():
                words[word]['count'] += data['count']
                # Limit to 5 example sentences to keep file sizes reasonable
                if len(words[word]['sentences']) < 5:
                    new_sentences = [s for s in data['sentences'] if s not in words[word]['sentences']]
                    words[word]['sentences'].extend(new_sentences)
                    words[word]['sentences'] = words[word]['sentences'][:5]
            grammar.update(chunk_grammar)

    print("\n  Parallel analysis complete.")
    return dict(words), grammar


def get_dictionary_priority(folder_path: str) -> List[str]:
    """Prompts the user to select and prioritize dictionary files."""
    if not os.path.exists(folder_path):
        print(f"Warning: Dictionary folder '{folder_path}' not found.")
        return []

    dictionaries = [f for f in os.listdir(folder_path) if f.endswith('.zip') or f.endswith('.json')]
    if not dictionaries:
        print(f"No dictionary files found in the '{folder_path}' folder.")
        return []

    print("\nFound the following dictionaries:")
    for i, name in enumerate(dictionaries):
        print(f"  [{i + 1}] {name}")

    while True:
        try:
            priority_input = input(f"Enter the desired order (e.g., '2,1'), or press Enter to skip: ")
            if not priority_input:
                return []
            priority_indices = [int(p.strip()) - 1 for p in priority_input.split(',')]
            
            if all(0 <= i < len(dictionaries) for i in priority_indices):
                return [dictionaries[i] for i in priority_indices]
            else:
                print("Invalid input. Please enter numbers corresponding to the dictionaries listed.")
        except (ValueError, IndexError):
            print("Invalid input format. Please enter comma-separated numbers (e.g., 2,1).")


def load_single_dictionary_job(filepath: str) -> Tuple[str, Dict[str, Any]]:
    """Loads a single dictionary file (zip or json) into memory."""
    dict_name = os.path.splitext(os.path.basename(filepath))[0]
    dictionary = {}
    
    def process_entry(entry):
        try:
            # Assumes a specific format, but handles failure gracefully
            term, _, _, _, _, definition, *_ = entry
            # Ensure definition is stored as a UTF-8 string
            if isinstance(definition, (list, dict)):
                return term, orjson.dumps(definition).decode('utf-8')
            return term, definition
        except (ValueError, TypeError):
            # print(f"WARNING: Malformed entry in '{dict_name}': {entry}") # Uncomment for debugging
            return None, None

    try:
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as z:
                for json_file in z.namelist():
                    if json_file.startswith('term_bank_') and json_file.endswith('.json'):
                        with z.open(json_file) as f:
                            data = orjson.loads(f.read())
                            for entry in data:
                                term, definition = process_entry(entry)
                                if term: dictionary[term] = definition
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = orjson.loads(f.read())
                for entry in data:
                    term, definition = process_entry(entry)
                    if term: dictionary[term] = definition
    except Exception as e:
        print(f"\nERROR: Failed to load dictionary '{filepath}': {e}")
        return dict_name, {}
        
    return dict_name, dictionary


def load_dictionaries_with_cache(folder_path: str, prioritized_list: List[str]) -> Dict[str, Dict]:
    """Loads dictionaries from source or cached .pkl files for speed."""
    dictionaries = {}
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    
    files_to_load_from_source = []
    
    # First, check for cache availability and user preference
    cached_files_found = {
        os.path.splitext(fname)[0]: os.path.join(CACHE_FOLDER, f"{os.path.splitext(fname)[0]}.pkl")
        for fname in prioritized_list if os.path.exists(os.path.join(CACHE_FOLDER, f"{os.path.splitext(fname)[0]}.pkl"))
    }
    
    use_cache = False
    if cached_files_found:
        prompt = f"Cache files found for {', '.join(cached_files_found.keys())}. Use them? (y/n): "
        if input(prompt).lower() == 'y':
            use_cache = True

    # Decide which files to load from source vs. cache
    for filename in prioritized_list:
        dict_name = os.path.splitext(filename)[0]
        if use_cache and dict_name in cached_files_found:
            print(f"  Loading '{dict_name}' from cache...")
            try:
                with open(cached_files_found[dict_name], 'rb') as f:
                    dictionaries[dict_name] = pickle.load(f)
            except Exception as e:
                print(f"  ERROR: Failed to load cache for '{dict_name}': {e}. Loading from source instead.")
                files_to_load_from_source.append(os.path.join(folder_path, filename))
        else:
            files_to_load_from_source.append(os.path.join(folder_path, filename))

    # Load any remaining files from their source
    if files_to_load_from_source:
        print("Loading dictionaries from source files...")
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(load_single_dictionary_job, filepath) for filepath in files_to_load_from_source}
            for i, future in enumerate(as_completed(futures), 1):
                dict_name, loaded_dict = future.result()
                dictionaries[dict_name] = loaded_dict
                sys.stdout.write(f"\r  Loaded {i}/{len(files_to_load_from_source)}: '{dict_name}' ({len(loaded_dict)} entries)")
                sys.stdout.flush()

                # Save the newly loaded dictionary to cache
                cache_file = os.path.join(CACHE_FOLDER, f"{dict_name}.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump(loaded_dict, f)
        print("\nDictionaries loaded and cached.")
        
    # Ensure the final dictionary order matches the user's priority
    return {name: dictionaries[name] for name in [os.path.splitext(p)[0] for p in prioritized_list] if name in dictionaries}


def add_definitions(words: Dict[str, Dict], dictionaries: Dict[str, Dict]) -> Dict[str, Dict]:
    """Adds definitions to words from the prioritized list of dictionaries."""
    print("Adding definitions to words...")
    for word in words:
        definitions = []
        # Iterate through dictionaries in their prioritized order
        for dict_name, dictionary in dictionaries.items():
            if word in dictionary:
                # Combine definitions from multiple dictionaries
                definitions.append(f"<b>{dict_name}:</b> {dictionary[word]}")
        words[word]['definitions'] = "".join(definitions)
    return words


def interactive_filter(words: Dict[str, Dict]) -> Dict[str, Dict]:
    """Applies a series of user-defined filters to the word list."""
    print("\n--- Interactive Filtering ---")
    if not words:
        print("  Word list is empty, nothing to filter.")
        return {}

    if input("Remove words without any definitions? (y/n): ").lower() == 'y':
        words = {w: d for w, d in words.items() if d.get('definitions')}
        print(f"  Words remaining: {len(words)}")

    try:
        min_freq = int(input("Enter minimum appearance count (e.g., 5, or 0 for no limit): "))
        if min_freq > 0:
            words = {w: d for w, d in words.items() if d['count'] >= min_freq}
            print(f"  Words remaining: {len(words)}")
    except ValueError:
        print("  Invalid number, skipping frequency filter.")

    try:
        sorted_words = sorted(words.items(), key=lambda item: item[1]['count'], reverse=True)
        top_n = int(input(f"Show how many top common words for manual exclusion? (e.g., 20): "))
        if top_n > 0:
            print("Top common words:")
            to_exclude_preview = sorted_words[:min(top_n, len(sorted_words))]
            for word, data in to_exclude_preview:
                print(f"  - {word} (appears {data['count']} times)")
            
            if input("Filter out these top N words? (y/n): ").lower() == 'y':
                to_remove = {word for word, data in to_exclude_preview}
                words = {w: d for w, d in words.items() if w not in to_remove}
                print(f"  Words remaining: {len(words)}")
    except ValueError:
        print("  Invalid number, skipping common word filter.")

    manual_exclude = input("Enter any other words to exclude (comma-separated): ")
    if manual_exclude:
        to_remove = {w.strip() for w in manual_exclude.split(',')}
        words = {w: d for w, d in words.items() if w not in to_remove}
        print(f"  Words remaining: {len(words)}")
        
    return words


def generate_anki_deck(words: Dict[str, Any], filename: str):
    """Generates a TSV file for Anki import from the final word list."""
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        # Sort by frequency for the final deck
        sorted_words = sorted(words.items(), key=lambda item: item[1]['count'], reverse=True)
        for word, data in sorted_words:
            formatted_sentences = '<ul>' + ''.join([f'<li>{s}</li>' for s in data.get('sentences', [])]) + '</ul>'
            # The new, robust function is called here to clean the definition
            formatted_definition = parse_definition(data.get('definitions', ''))
            writer.writerow([word, formatted_definition, formatted_sentences, data.get('count', 0)])


def process_file(file_path: str) -> Tuple[Dict, Counter]:
    """Orchestrates the full analysis pipeline for a single file."""
    print(f"\n--- Processing file: {os.path.basename(file_path)} ---")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    text = extract_text_from_epub(file_path) if file_extension == '.epub' else extract_text_from_txt(file_path)
    
    if not text:
        return {}, Counter()

    words, grammar = analyze_korean_text_parallel(text)
    print(f"  Found {len(words)} unique words.")
    
    return words, grammar


def main():
    """Main execution function."""
    source_files = find_source_files()
    if not source_files:
        print("Error: No .epub or .txt files found in '.', 'input/', or 'novels/' directories.")
        return

    selected_files = []
    if len(source_files) > 1:
        print("Multiple files found:")
        for i, filename in enumerate(source_files, 1):
            print(f"  [{i}] {filename}")
        
        choice = input("Enter 'all' or comma-separated numbers (e.g., '1,3') to process: ").lower()
        if choice == 'all':
            selected_files = source_files
        else:
            try:
                indices = [int(i.strip()) - 1 for i in choice.split(',')]
                selected_files = [source_files[i] for i in indices if 0 <= i < len(source_files)]
            except (ValueError, IndexError):
                print("Invalid input. Exiting.")
                return
    else:
        selected_files = source_files

    if not selected_files:
        print("No files selected. Exiting.")
        return

    # Load dictionaries
    prioritized_dictionaries = get_dictionary_priority(DICTIONARIES_FOLDER)
    dictionaries = {}
    if prioritized_dictionaries:
        dictionaries = load_dictionaries_with_cache(DICTIONARIES_FOLDER, prioritized_dictionaries)

    all_words_combined = defaultdict(lambda: {'count': 0, 'sentences': [], 'definitions': ''})

    # Process each selected file
    for file_path in selected_files:
        words, grammar = process_file(file_path)
        if not words:
            continue

        if dictionaries:
            words = add_definitions(words, dictionaries)
        
        # --- Generate per-book output files ---
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs(JSON_FOLDER, exist_ok=True)
        os.makedirs(GRAMMAR_FOLDER, exist_ok=True)
        os.makedirs(ANKI_FOLDER, exist_ok=True)
        
        # Save JSON analysis
        with open(os.path.join(JSON_FOLDER, f'{base_name}_word_analysis.json'), 'wb') as f:
            f.write(orjson.dumps(words, option=orjson.OPT_INDENT_2))
        
        # Save grammar analysis
        with open(os.path.join(GRAMMAR_FOLDER, f'{base_name}_grammar_analysis.txt'), 'w', encoding='utf-8') as f:
            for item, count in grammar.most_common():
                f.write(f"{item}: {count}\n")
        
        # Generate per-book Anki deck
        generate_anki_deck(words, os.path.join(ANKI_FOLDER, f'{base_name}_anki_deck.csv'))
        print(f"  Outputs for '{base_name}' generated in the 'output/' directory.")

        # --- Merge results for combined output ---
        for word, data in words.items():
            all_words_combined[word]['count'] += data['count']
            if not all_words_combined[word]['definitions']:
                all_words_combined[word]['definitions'] = data.get('definitions', '')
            if len(all_words_combined[word]['sentences']) < 5:
                new_sentences = [s for s in data['sentences'] if s not in all_words_combined[word]['sentences']]
                all_words_combined[word]['sentences'].extend(new_sentences)
                all_words_combined[word]['sentences'] = all_words_combined[word]['sentences'][:5]

    # --- Generate combined output if multiple files were processed ---
    if len(selected_files) > 1:
        print("\n--- Generating combined output for all processed files ---")
        # Apply interactive filter to the combined set of words
        filtered_combined_words = interactive_filter(dict(all_words_combined))
        
        combined_anki_file = os.path.join(ANKI_FOLDER, 'combined_anki_deck.csv')
        generate_anki_deck(filtered_combined_words, combined_anki_file)
        print(f"  - Combined Anki deck saved to: {combined_anki_file}")

    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    # Required for multiprocessing to work correctly on some platforms (like Windows)
    import multiprocessing
    multiprocessing.freeze_support()
    main()
