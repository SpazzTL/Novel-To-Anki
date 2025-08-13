import json
import re
import os
import csv
from collections import Counter
import ebooklib
from ebooklib import epub
from konlpy.tag import Okt
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import sys
from glob import glob
from typing import Dict, Any

# --- New functions from process_anki_data.py for formatting ---
def parse_definition(definition_text: str) -> str:
    """
    Parses the complex definition text to extract meanings and examples.
    """
    html_output = '<div class="definition-content">'
    
    # Check if the definition text contains the complex JSON-like structure
    if 'KRDICT-EN-hanja.v1.0.3-hanja-only' in definition_text:
        try:
            # Extract the relevant JSON-like part
            json_part = re.search(r"KRDICT-EN-hanja\.v1\.0\.3-hanja-only:\s*(\[.*\])", definition_text, re.DOTALL).group(1)
            # Fix single quotes and other JSON-incompatibilities
            json_part = json_part.replace("'", '"').replace("`", '"')
            data = json.loads(json_part)
            
            structured_content = data[0]['content']
            
            meaning_sections = [c for c in structured_content if isinstance(c, dict) and 'style' in c and 'fontWeight' in c['style'] and 'content' in c and c['content'].isdigit()]
            
            if meaning_sections:
                for section in meaning_sections:
                    meaning_number = section['content']
                    meaning_index = structured_content.index(section)
                    
                    english_meaning = structured_content[meaning_index + 1]['content'] if meaning_index + 1 < len(structured_content) and 'content' in structured_content[meaning_index + 1] else 'No meaning found'
                    
                    description_text = structured_content[meaning_index + 2]['content'] if meaning_index + 2 < len(structured_content) and 'content' in structured_content[meaning_index + 2] else ''
                    
                    examples_list = structured_content[meaning_index + 3]['content'] if meaning_index + 3 < len(structured_content) and 'tag' in structured_content[meaning_index + 3] and structured_content[meaning_index + 3]['tag'] == 'ul' else []
                    
                    html_output += f'<div class="definition-section">'
                    html_output += f'<p class="definition-meaning"><b>Meaning {meaning_number}:</b> {english_meaning}</p>'
                    if description_text:
                        html_output += f'<p class="definition-desc">{description_text}</p>'
                    
                    if examples_list:
                        html_output += '<ul class="definition-examples">'
                        for example in examples_list:
                            html_output += f'<li>{example["content"]}</li>'
                        html_output += '</ul>'
                    html_output += '</div>'

            html_output += '</div>'
            return html_output

        except (json.JSONDecodeError, IndexError, AttributeError):
            # Fallback to plain text if parsing fails
            return f'<div class="definition-content">{definition_text}</div>'
    else:
        # Fallback for simpler definitions
        return f'<div class="definition-content">{definition_text}</div>'

# --- Existing functions from epub-anki.py (with modifications) ---
def find_epub_files():
    """Finds all .epub files in the current directory and returns a list."""
    return glob('*.epub')

def extract_text_from_epub(epub_path):
    """Extracts text content from an EPUB file."""
    book = epub.read_epub(epub_path)
    text_content = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        content = item.get_body_content().decode('utf-8', 'ignore')
        text_content.append(clean_html(content))
    return "".join(text_content)

def clean_html(raw_html):
    """Removes HTML tags and extra whitespace from a string."""
    cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(cleanr, '', raw_html)
    return re.sub(r'\s+', ' ', cleantext).strip()

def get_korean_sentences(text):
    """Splits text into a list of Korean sentences."""
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def analyze_sentence(okt, sentence):
    """Analyzes a single sentence to extract words and grammar patterns."""
    words = {}
    grammar = []
    try:
        pos_tagged = okt.pos(sentence, norm=True, stem=True)
        for word, pos in pos_tagged:
            if pos in ['Noun', 'Verb', 'Adjective', 'Adverb'] and len(word) > 1:
                if word not in words:
                    words[word] = {'count': 0, 'sentences': [], 'definitions': {}}
                words[word]['count'] += 1
                if sentence not in words[word]['sentences']:
                    words[word]['sentences'].append(sentence)

            if pos not in ['Punctuation', 'Foreign']:
                grammar.append(f"{word}/{pos}")
    except Exception:
        pass  # Ignore sentences that cause errors
    return words, grammar

def analyze_korean_text_threaded(text):
    """Analyzes Korean text using multiple threads."""
    sentences = get_korean_sentences(text)
    words = {}
    grammar = Counter()
    
    print("  Initializing NLP analyzer (Okt)...")
    okt = Okt()
    print("  Analyzer initialized. Starting threaded analysis...")

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(analyze_sentence, okt, sentence) for sentence in sentences}
        
        for i, future in enumerate(as_completed(futures)):
            print(f"\r  Analyzing sentences: {i + 1}/{len(sentences)}", end="")
            
            sentence_words, sentence_grammar = future.result()
            for word, data in sentence_words.items():
                if word not in words:
                    words[word] = {'count': 0, 'sentences': [], 'definitions': {}}
                words[word]['count'] += data['count']
                for sent in data['sentences']:
                    if sent not in words[word]['sentences'] and len(words[word]['sentences']) < 5:
                        words[word]['sentences'].append(sent)
            grammar.update(sentence_grammar)
    
    print("\n  Threaded analysis complete.")
    return words, grammar

def get_dictionary_priority(folder_path):
    """Prompts the user to set the priority for dictionary files."""
    if not os.path.exists(folder_path):
        print(f"Warning: Dictionary folder '{folder_path}' not found.")
        return []

    dictionaries = [f for f in os.listdir(folder_path) if f.endswith('.zip')]
    if not dictionaries:
        print("No dictionary files found.")
        return []

    print("Found the following dictionaries:")
    for i, name in enumerate(dictionaries):
        print(f"  [{i + 1}] {name}")

    while True:
        try:
            priority_input = input(f"\nEnter the desired order (e.g., '3,1,2'): ")
            priority_indices = [int(p.strip()) - 1 for p in priority_input.split(',')]
            
            if all(0 <= i < len(dictionaries) for i in priority_indices) and len(priority_indices) == len(dictionaries):
                return [dictionaries[i] for i in priority_indices]
            else:
                print("Invalid input. Please enter a comma-separated list of numbers corresponding to the dictionaries.")
        except (ValueError, IndexError):
            print("Invalid input format.")

def load_single_dictionary_job(zip_path):
    """A helper function to load a single dictionary from a ZIP archive."""
    dict_name = os.path.splitext(os.path.basename(zip_path))[0]
    dictionary = {}
    try:
        try:
            import orjson
            is_orjson = True
        except ImportError:
            import json
            is_orjson = False

        with zipfile.ZipFile(zip_path, 'r') as z:
            for json_file in z.namelist():
                if json_file.startswith('term_bank_') and json_file.endswith('.json'):
                    with z.open(json_file) as f:
                        if is_orjson:
                            data = orjson.loads(f.read())
                        else:
                            data = json.load(f)

                        for entry in data:
                            term = entry[0]
                            definition = entry[5] if len(entry) > 5 else 'No definition found'
                            dictionary[term] = definition
    except Exception as e:
        print(f"An error occurred with dictionary '{zip_path}': {e}")
        return dict_name, {}
    return dict_name, dictionary

def load_dictionaries_with_cache(folder_path, prioritized_list, cache_folder='cache'):
    """
    Loads dictionaries using individual cache files for each dictionary.
    """
    dictionaries = {}
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    zip_paths_to_load_from_source = []
    cached_dictionaries_to_load = {}
    for filename in prioritized_list:
        dict_name = os.path.splitext(filename)[0]
        cache_file = os.path.join(cache_folder, f"{dict_name}.pkl")
        if os.path.exists(cache_file):
            cached_dictionaries_to_load[dict_name] = cache_file
        else:
            zip_paths_to_load_from_source.append(os.path.join(folder_path, filename))

    if cached_dictionaries_to_load:
        cached_names = ', '.join(cached_dictionaries_to_load.keys())
        use_cache = input(f"Cache files found for {cached_names}. Use cached versions? (y/n): ")
        if use_cache.lower() == 'y':
            for dict_name, cache_file in cached_dictionaries_to_load.items():
                print(f"  Loading '{dict_name}' from cache...")
                try:
                    with open(cache_file, 'rb') as f:
                        dictionaries[dict_name] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading cache for '{dict_name}': {e}. Will load from source instead.")
                    zip_paths_to_load_from_source.append(os.path.join(folder_path, f"{dict_name}.zip"))
    
    if zip_paths_to_load_from_source:
        print("Loading dictionaries from source...")
        total_dictionaries = len(zip_paths_to_load_from_source)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(load_single_dictionary_job, zip_path) for zip_path in zip_paths_to_load_from_source}
            
            for i, future in enumerate(as_completed(futures)):
                dict_name, loaded_dict = future.result()
                dictionaries[dict_name] = loaded_dict
                
                sys.stdout.write(f"\r  Loading dictionary {i + 1}/{total_dictionaries}: '{dict_name}'")
                sys.stdout.flush()
                
                cache_file = os.path.join(cache_folder, f"{dict_name}.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump(loaded_dict, f)
        print("\nAll dictionaries loaded.")
    elif not dictionaries:
        print("\nSkipping dictionary loading.")
        return {}

    return dictionaries

def add_definitions(words, dictionaries):
    """Adds multiple English definitions to the word data."""
    for word in words:
        definitions = []
        for dict_name, dictionary in dictionaries.items():
            if word in dictionary:
                definitions.append(f"<b>{dict_name}:</b> {dictionary[word]}")
        words[word]['definitions'] = "<br>".join(definitions)
    return words

def interactive_filter(words):
    """Guides the user through filtering the word list."""
    print("\n--- Interactive Filtering ---")
    
    if input("Remove words without definitions? (y/n): ").lower() == 'y':
        words = {w: d for w, d in words.items() if d.get('definitions')}
        print(f"  Words remaining: {len(words)}")

    try:
        min_freq = int(input("Enter minimum appearance count (e.g., 3). Enter 0 for no limit: "))
        if min_freq > 0:
            words = {w: d for w, d in words.items() if d['count'] >= min_freq}
            print(f"  Words remaining: {len(words)}")
    except ValueError:
        print("  Invalid number. Skipping frequency filter.")
        
    try:
        sorted_words = sorted(words.items(), key=lambda item: item[1]['count'], reverse=True)
        top_n = int(input(f"How many of the most common words to show for exclusion? (e.g., 20, max {len(sorted_words)}): "))
        if top_n > 0:
            print("Top common words:")
            for i in range(min(top_n, len(sorted_words))):
                print(f"  - {sorted_words[i][0]} (appears {sorted_words[i][1]['count']} times)")
            
            if input("Filter out these top N most common words? (y/n): ").lower() == 'y':
                to_remove = {w[0] for w in sorted_words[:top_n]}
                words = {w: d for w, d in words.items() if w not in to_remove}
                print(f"  Words remaining: {len(words)}")
                
    except ValueError:
        print("  Invalid number. Skipping common word filter.")

    manual_exclude = input("Enter any other words to exclude, separated by commas (e.g., 하다,가다): ")
    if manual_exclude:
        to_remove = {w.strip() for w in manual_exclude.split(',')}
        words = {w: d for w, d in words.items() if w not in to_remove}
        print(f"  Words remaining: {len(words)}")

    return words

def generate_anki_deck(words: Dict[str, Any], filename="anki_deck.csv"):
    """
    Generates a CSV file for Anki, with pre-formatted sentences and definitions.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Sort words by frequency (count) in descending order
        sorted_words = sorted(words.items(), key=lambda item: item[1]['count'], reverse=True)
        
        for word, data in sorted_words:
            # Process sentences: format as an HTML unordered list
            sentences_list = data.get('sentences', [])
            formatted_sentences = '<ul>' + ''.join([f'<li>{s}</li>' for s in sentences_list]) + '</ul>'

            # Process definition: use the new parsing function
            raw_definition = data.get('definitions', 'No definition found.')
            formatted_definition = parse_definition(raw_definition)
            
            # This is the updated line to change the output order
            writer.writerow([
                word,
                formatted_definition,
                formatted_sentences,
                data.get('count', 0)
            ])

def main():
    epub_files = find_epub_files()
    if not epub_files:
        print("Error: No .epub file found in the current directory.")
        return

    selected_files = []
    if len(epub_files) > 1:
        print("Multiple EPUB files found:")
        for i, filename in enumerate(epub_files):
            print(f"  [{i+1}] {filename}")
        
        choice = input("Enter 'all' to process all, or a comma-separated list of numbers (e.g., '1,3'): ").lower()
        if choice == 'all':
            selected_files = epub_files
        else:
            try:
                indices = [int(i.strip()) - 1 for i in choice.split(',')]
                selected_files = [epub_files[i] for i in indices if 0 <= i < len(epub_files)]
            except (ValueError, IndexError):
                print("Invalid input. Processing all files.")
                selected_files = epub_files
    else:
        selected_files = epub_files

    if not selected_files:
        print("No files selected for processing. Exiting.")
        return

    dictionaries_folder = 'Dictionaries'
    output_folder = 'output'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    prioritized_dictionaries = get_dictionary_priority(dictionaries_folder)
    dictionaries = {}
    if prioritized_dictionaries:
        print("\nLoading dictionaries...")
        dictionaries = load_dictionaries_with_cache(dictionaries_folder, prioritized_dictionaries)

    all_words_combined = {}

    for epub_file in selected_files:
        print(f"\n--- Processing EPUB file: {epub_file} ---")
        
        base_name = os.path.splitext(epub_file)[0]
        output_json_file = os.path.join(output_folder, f'{base_name}_word_analysis.json')
        grammar_output_file = os.path.join(output_folder, f'{base_name}_grammar_analysis.txt')
        anki_csv_file = os.path.join(output_folder, f'{base_name}_anki_deck.csv')

        text = extract_text_from_epub(epub_file)

        print("\nAnalyzing Korean text (this may take a while)...")
        words, grammar = analyze_korean_text_threaded(text)
        print(f"Found {len(words)} unique words.")

        if dictionaries:
            print("Adding definitions to words.")
            words = add_definitions(words, dictionaries)

        filtered_words = interactive_filter(words)
        
        print("\nGenerating output files for this book...")
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_words, f, ensure_ascii=False, indent=4)
            
        with open(grammar_output_file, 'w', encoding='utf-8') as f:
            for item, count in grammar.most_common():
                f.write(f"{item}: {count}\n")
                
        generate_anki_deck(filtered_words, anki_csv_file)
        
        for word, data in filtered_words.items():
            if word not in all_words_combined:
                all_words_combined[word] = data
            else:
                all_words_combined[word]['count'] += data['count']
                all_words_combined[word]['sentences'].extend(data['sentences'])

    if len(selected_files) > 1:
        print("\n--- Generating combined output ---")
        combined_anki_file = os.path.join(output_folder, 'combined_anki_deck.csv')
        generate_anki_deck(all_words_combined, combined_anki_file)
        print(f"  - Combined Anki deck CSV saved to: {combined_anki_file}")
    
    print(f"\nAnalysis complete!")

if __name__ == '__main__':
    main()