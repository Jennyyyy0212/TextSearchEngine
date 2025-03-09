from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import os,mmap, glob, math
from inverted_index import process_text, build_inverted_output, print_merged_index



stemmer = PorterStemmer()
stopwords = set(stopwords.words('english'))


class SearchEngine():
    def __init__(self,input_dir):
        self.index =  defaultdict(lambda: defaultdict(lambda: {'freq': 0, 'positions': []}))
        self.input_dir = input_dir
        self.output_dir = "inv-index"
        self._documents = {}
        self.total_document_count = self.count_total_document(input_dir)

    def count_total_document(self, input_dir):
        file_count = sum(len([file for file in files if file.endswith('.zip')]) for _, _, files in os.walk(input_dir))
        return file_count

    def load_shard(self, term):
        shard_file = os.path.join(self.output_dir, f"{term[0].lower()}.txt")  # Shard by the first letter
        if os.path.exists(shard_file):
            line = self.find_term_mmap(term, shard_file)
            parts = line.strip().split()
            current_word = parts[0] if parts else None
            if current_word == term:
                # Found the term, process it
                doc_info = parts[1].split(';')
                for doc in doc_info:
                    doc_id, freq, positions = doc.split(':')
                    freq = int(freq)
                    positions = list(map(int, positions.split(',')))
                    self.index[term][doc_id] = {'freq': freq, 'positions': positions}
                return
        else:
            print(f"Shard file for '{term[0]}.txt' not found.")


    def find_term_mmap(self, term, shard_file):
        with open(shard_file, 'r+b') as file:
            with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for line in iter(mm.readline, b""):
                    line = line.decode('utf-8').strip()
                    if line.startswith(term):
                        return line
        return None

    def load_index_for_query(self, query_terms):
        # Clear the index for a new query
        self.index.clear()
        # Load only the shards needed for the query terms
        unique_term = set(query_terms)
        for term in unique_term: #make sure no duplicate term in the index 
            self.load_shard(term)
    
    def retrieve_documents(self, query_word_list, n_result):
        document_scores = {}

        # Calculate tf-idf for query terms
        query_tf_idf = self.calculate_query_tf_idf(query_word_list)

        # Get all documents referenced by the query terms
        referenced_docs = set()
        for term in query_word_list:
            if term in self.index:
                referenced_docs.update(self.index[term].keys())

        # Process each document
        for doc in referenced_docs:
            # Initialize score
            document_scores[doc] = 0

            # Calculate cosine similarity for the document
            cosine_similarity = self.calculate_cosine_similarity(doc, query_tf_idf)

            # Calculate position score for the document
            position_score = 0
            for term in query_word_list:
                position_score += self.calculate_position(term, query_word_list, doc)

            # Update cumulative score
            document_scores[doc] = cosine_similarity + position_score
    
        return len(document_scores), sorted(document_scores.items(), key=lambda x: x[1], reverse=True)[:n_result]
    
    def calculate_query_tf_idf(self, query_terms):
        """Calculate tf-idf for weight for query terms."""
        tf_idf = {}
        unique_terms = set(query_terms)  # Use a set to ensure terms are unique
        for term in unique_terms:
            tf = 1 + math.log10(query_terms.count(term))
            idf = self.calculate_idf(term)
            tf_idf[term] = tf * idf
        return tf_idf
    
    def calculate_tf_idf(self, term, doc):
        """Calculate tf-idf for weight for a term in a document."""
        tf = 1 + math.log10(self.index[term][doc]['freq'])
        idf = self.calculate_idf(term)
        return tf * idf
    
    def calculate_idf(self, term):
        """Calculate inverse document frequency for a term."""
        doc_count = len(self.index[term])
        # Total number of documents in the corpus - self.total_document_count
        return math.log10(self.total_document_count / doc_count)
    
    def calculate_cosine_similarity(self, doc, query_tf_idf):
        """Calculate the cosine similarity between query and document tf-idf vectors."""
        doc_tf_idf = {term: self.calculate_tf_idf(term, doc) for term in self.index if doc in self.index[term]}
        # calcualte the document's tf-idf score for those term in index (where sum up all the terms in the query and each term will show up once in local index dic)

        # Compute dot product and magnitudes for cosine similarity
        dot_product = sum(query_tf_idf.get(term, 0) * doc_tf_idf.get(term, 0) for term in self.index)
        query_magnitude = math.sqrt(sum(val ** 2 for val in query_tf_idf.values()))
        doc_magnitude = math.sqrt(sum(val ** 2 for val in doc_tf_idf.values()))
        
        # Handle cases where magnitude is zero
        if query_magnitude == 0 or doc_magnitude == 0:
            return 0
        return dot_product / (query_magnitude * doc_magnitude)
    
    def calculate_position(self, term, query_terms, doc):
        """Calculate the proximity score for the current term based on its position relative to other query terms."""
        position_score = 0

        # Get positions of the current term
        current_positions = self.index[term][doc]['positions']

        # Compare with all other query terms
        for other_term in query_terms:
            if other_term == term:
                continue
            if other_term in self.index and doc in self.index[other_term]:
                other_positions = self.index[other_term][doc]['positions']
                # Two-pointer traversal for sorted positions
                i, j = 0, 0
                while i < len(current_positions) and j < len(other_positions):
                    # Calculate distance and update score
                    distance = abs(current_positions[i] - other_positions[j])
                    position_score += 1 / distance if distance > 0 else 0
                    # Move the pointer with the smaller position
                    if current_positions[i] < other_positions[j]:
                        i += 1
                    else:
                        j += 1

        # Normalize the position score
        if len(query_terms) > 1:
            position_score /= (len(query_terms) - 1)
        return position_score
    


    def find_zip_path(self, doc_name):
        # Find the zip file within input_dir, replacing .txt with .zip
        file_name = doc_name.replace('.txt', '') # Can also replace the name only when written in the inverted index
        file_path = None

        # Search recursively for the zip file in the input_dir and subdirectories
        # bc we have known that each dir will only have one zipped file in the assumption so no need to list the file path out
        # if want to search for same name file, use f"{file_name}.*" for any file type
        for path in glob.glob(os.path.join(self.input_dir, '**', f"{file_name}"), recursive=True): 
            if os.path.isdir(path): #use os.path.isfile(path) to search for files 
                file_path = path
                break
        
        # Get the relative path if the file was found
        if file_path:
            return os.path.relpath(file_path, start=self.input_dir)
        else:
            print(f"Zip file for {doc_name} not found in input_dir.")
            return None
    
    def run(self):
        print("Search interface initialized. Enter queries or type 'exit' to quit.")
        build_inverted_output(self.input_dir,self.output_dir)
        while True:
            query = input("Type Query:  ")
            if query.lower() == 'exit':
                break
            query_words_list = process_text(query)
            self.load_index_for_query(query_words_list)
            print(f"{len(self.index)} of keywords") #number of inverted index will look at
            n_doc, results = self.retrieve_documents(query_words_list, n_result= 10)
            print(f"{n_doc} of total matching documents")
            print("Top 10 results:")
            for rank, (doc, score) in enumerate(results, 1):
                zip_file_path = self.find_zip_path(doc)
                # Check if the file was found and construct the relative URL
                if zip_file_path:
                    # Format it as a URL-like path
                    formatted_url = zip_file_path.replace(os.sep, '/')
                    print(f"{rank}. {formatted_url}") #score for debugging
                    #print(f"{rank}. {formatted_url} - Score: {score}") #score for debugging
                else: 
                    print(f"{rank}. File {zip_file_path} not found")

if __name__ == "__main__":
    input_dir = "input-files"
    search_interface = SearchEngine(input_dir)
    search_interface.run()
    #search_interface.load_index_for_query(["cabl","cage"],output_dir)
    #print(print_merged_index(search_interface.index))

    
