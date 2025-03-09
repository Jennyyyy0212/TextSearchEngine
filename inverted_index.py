from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import nltk, re, string,os, zipfile
import threading, os, time, shutil
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))

def process_text(text):
    # Tokenize by removing non-alphanumeric characters and converting to lowercase
    removed = text.translate(str.maketrans('', '', string.punctuation)) #remove all punctuation
    tokens = word_tokenize(removed)
     # Remove stop words and stem the text
    filtered_words = [stemmer.stem(token) for token in tokens if token.lower() not in stopwords]
    return filtered_words

def process_one_file(input_file):
    with open(input_file, 'r', encoding='ascii', errors='ignore') as f:
        text = f.read()
    transformed_text = process_text(text)
    # also use the file name only without the extension by use: document_name = os.path.splitext(document_name_with_extension)[0]
    document_name = os.path.basename(input_file)
    index_data = defaultdict(lambda: {'freq': 0, 'positions': []})

    for position, word in enumerate(transformed_text, start =1): #??
        index_data[word]['freq'] += 1
        index_data[word]['positions'].append(position)

    return document_name,index_data

def process_files(file_queue):
    local_inverted_index = defaultdict(lambda: defaultdict(lambda: {'freq': 0, 'positions': []}))
    while True:
        file_path = file_queue.get() #Remove and return an item from the queue
        if file_path is None:  # Signal to end processing
            file_queue.put(None)  # Pass Signal to other threads
            break
        
        # Process the file and update the inverted index
        document_name, file_index_data = process_one_file(file_path)
        for word, data in file_index_data.items():
            local_inverted_index[word][document_name]['freq'] += data['freq']
            local_inverted_index[word][document_name]['positions'].extend(data['positions'])
        
        file_queue.task_done()  #.task_done lets workers say when a task is done
        # For each get() used to fetch a task, a subsequent call to task_done() tells the queue that the process
        # If a join() is currently blocking, it will resume when all items have been processed (meaning that a task_done() call was received for every item that had been put() into the queue).
    return local_inverted_index

def merge_indexes(global_index, local_index):
    for word, docs in local_index.items():
        for doc_name, data in docs.items():
            # Directly assign each document's data to global_index without aggregation
            global_index[word][doc_name] = {'freq': data['freq'], 'positions': data['positions']}

def print_index_data(index_data):
    for word, data in index_data.items():
        print(word, "- Freq:", data['freq'], "at", ','.join(map(str, data['positions'])))

# Function to print merged index results for easy verification
def print_merged_index(index):
    for word, docs in index.items():
        print(f"Word: {word}")
        for doc_name, data in docs.items():
            print(f"  Document: {doc_name}, Frequency: {data['freq']}, Positions: {data['positions']}")

def cleanup_temp_dir(temp_dir):
    """Remove all files and directories within the temp directory."""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def unzip_files(input_dir,queue,temp_dir="temp_unzip_dir"):
    # Ensure the output directory is empty and exists
    if os.path.exists(temp_dir):
        # Remove existing files and folders in output directory
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
    else:
        os.makedirs(temp_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.endswith(".zip"):
                        zip_path = os.path.join(root, file)
                        # Extract files from the zip archive
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            # Extract .txt files
                            text_files = [f for f in zip_ref.namelist() if f.endswith(".txt") and not f.startswith('__')]
                            for text_file in text_files:
                                extracted_path = zip_ref.extract(text_file, temp_dir)
                                queue.put(extracted_path)  # Add extracted file path to queue
    queue.put(None)  # Signal to indicate end of unzipping

def inverted(input_dir):
    global_inverted_index = defaultdict(lambda: defaultdict(lambda: {'freq': 0, 'positions': []}))
    file_queue = Queue()
    cpu_count = os.cpu_count() # Determine optimal max_workers
    # CPU-bound tasks
    max_workers_cpu = cpu_count or 1         # Fallback to 1 if cpu_count is None
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers_cpu) as exec:
        exec.submit(unzip_files, input_dir, file_queue)
        futures.append(exec.submit(process_files, file_queue))  

        for future in as_completed(futures):
            local_index = future.result()
            merge_indexes(global_inverted_index, local_index)
    return global_inverted_index

def shard_files(inverted_index, output_dir):
     # Partition terms by their first letter
    partitioned_words = {letter: {} for letter in string.ascii_lowercase}
    
    # term is the word
    for word in inverted_index.keys():
        first_letter = word[0].lower()
        if first_letter in partitioned_words:
            partitioned_words[first_letter][word] = inverted_index[word]
    
    # Sort and write each partitioned set of terms to their corresponding shard file
    for letter, words in partitioned_words.items():
        file_path = os.path.join(output_dir, f"{letter}.txt")
        with open(file_path, "w") as shard_file:
            # Sort words for the current letter
            for word in sorted(words.keys()):   #Sort terms alphabetically and write the index to disk
                postings = []
                for doc_name, data in words[word].items():
                    posting = f"{doc_name}:{data['freq']}:{','.join(map(str, data['positions']))}"
                    postings.append(posting)
                
                # Write sorted term and postings to the shard file
                shard_file.write(f"{word} {';'.join(postings)}\n")
    print(f"Inverted index sharded and written to directory: {output_dir}")
    
def build_inverted_output(input_dir, output_dir,temp_dir = "temp_unzip_dir"):
    # Check if output_dir exists, and create it if it doesn’t
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} created.")
    
    # Check if the directory is empty
    if any(f for f in os.listdir(output_dir) if not f.startswith('.')):
        print(f"{output_dir} already has an inverted index.")
        return
    inverted_index = inverted(input_dir)
    shard_files(inverted_index, output_dir)
    cleanup_temp_dir(temp_dir)
    return output_dir

if __name__ == "__main__":
    start_time = time.time()  # Start timer
    # build_inverted_output("test-input-files","test-inv-index")
    build_inverted_output("input-files","inv-index")
    # Total time
    end_time = time.time()
    print(f"Total runtime: {end_time - start_time:.2f} seconds")