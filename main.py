#-------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5180- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/
# importing required libraries
import pandas as pd
import heapq
from sklearn.feature_extraction.text import CountVectorizer
# -----------------------------
# PARAMETERS
# -----------------------------
INPUT_PATH = "corpus/corpus.tsv"
BLOCK_SIZE = 100
NUM_BLOCKS = 10
READ_BUFFER_LINES_PER_FILE = 100
WRITE_BUFFER_LINES = 500
# ---------------------------------------------------------
# 1) READ FIRST BLOCK OF 100 DOCUMENTS USING PANDAS
# ---------------------------------------------------------
# Use pandas.read_csv with chunksize=100.
# Each chunk corresponds to one memory block.
# Convert docIDs like "D0001" to integers.
# ---------------------------------------------------------
df = pd.read_csv(INPUT_PATH, sep="\t", chunksize=BLOCK_SIZE, header=None, converters={0: lambda x: int(x.lstrip("D"))})
# ---------------------------------------------------------
# 2) BUILD PARTIAL INDEX (SPIMI STYLE) FOR CURRENT BLOCK
# ---------------------------------------------------------
# - Use CountVectorizer(stop_words='english')
# - Fit and transform the 100 documents
# - Reconstruct binary postings lists from the sparse matrix
# - Store postings in a dictionary: term -> set(docIDs)
# ---------------------------------------------------------
count_vectorizer = CountVectorizer(stop_words='english')
for i, chunk in enumerate(df): # Iterate over each block (chunk) of 100 documents
    dictionary: dict[str, set[int]] = {}
    for _, row in chunk.iterrows(): # Iterate over each document in the block
        doc_ids = row[0]
        text = row[1]
        count_vectorizer.fit([text])
        terms = count_vectorizer.get_feature_names_out()
        for term in terms:
            if term not in dictionary:
                dictionary[term] = set()
            dictionary[term].add(doc_ids)

# ---------------------------------------------------------
# 3) FLUSH PARTIAL INDEX TO DISK
# ---------------------------------------------------------
# - Sort terms lexicographically
# - Sort postings lists (ascending docID)
# - Write to: block_1.txt, block_2.txt, ..., block_10.txt
# - Format: term:docID1,docID2,docID3
# ---------------------------------------------------------
    del chunk # Free memory for the current block
    with open(f"block_{i + 1}.txt", "w") as f:
        buffer = []
        for term in sorted(dictionary.keys()):
            postings_list = sorted(dictionary[term])
            postings_str = ",".join(f"{doc_id}" for doc_id in postings_list)
            buffer.append(f"{term}:{postings_str}\n")
            if len(buffer) >= WRITE_BUFFER_LINES:
                f.writelines(buffer)
                buffer.clear()
        if buffer:
            f.writelines(buffer) # Flush remaining buffer to disk
# ---------------------------------------------------------
# 5) FINAL MERGE PHASE
# ---------------------------------------------------------
# After all block files are created:
# - Open block_1.txt ... block_10.txt simultaneously
# ---------------------------------------------------------
block_files = [open(f"block_{i + 1}.txt", "r") for i in range(NUM_BLOCKS)]
# ---------------------------------------------------------
# 6) INITIALIZE READ BUFFERS
# ---------------------------------------------------------
# For each block file:
# - Read up to READ_BUFFER_LINES_PER_FILE lines
# - Parse each line into (term, postings_list)
# - Store in a per-file read buffer
# ---------------------------------------------------------
read_buffers: list[list[tuple[str, list[int]]]] = []
for block_file in block_files:
    buffer = []
    for _ in range(READ_BUFFER_LINES_PER_FILE):
        line = block_file.readline()
        if not line:
            break
        term, postings_str = line.strip().split(":")
        postings_list = list(map(int, postings_str.split(",")))
        buffer.append((term, postings_list))
    read_buffers.append(buffer)
# ---------------------------------------------------------
# 7) INITIALIZE MIN-HEAP (OR SORTED STRUCTURE)
# ---------------------------------------------------------
# - Push the first term from each read buffer into a min-heap
# - Heap elements: (term, file_index)
# ---------------------------------------------------------
min_heap: list[tuple[str, list[int], int]] = []
for i, buffer in enumerate(read_buffers):
    if buffer:
        term, doc_ids = buffer.pop(0)
        heapq.heappush(min_heap, (term, doc_ids, i))

# ---------------------------------------------------------
# 8) MERGE LOOP
# ---------------------------------------------------------
# While min-heap is not empty:
# 1. Pop the min-heap root (smallest term)
# 2. Keep popping the min-heap root while the current term equals the previous term
# 3. Collect all read buffers whose current term matches
# 4. Merge postings lists associated with this term (sorted + deduplicated)
# 5. Advance corresponding read buffer pointers
# 6. If a read buffer is exhausted, read next 100 lines from the corresponding block (if available)
# 7. For each read buffer whose pointer advanced, push its new pointed term into the heap (if available).
# ---------------------------------------------------------
with open("final_index.txt", "w") as final_index_file:
    write_buffer = []
    while len(min_heap) > 0:
        doc_ids_with_current_term = []
        current_term, doc_ids, read_buffer_index = heapq.heappop(min_heap)
        read_buffers_to_advance = [read_buffer_index]
        for doc_ids in doc_ids:
            doc_ids_with_current_term.append(doc_ids)
        while len(min_heap) > 0 and min_heap[0][0] == current_term:
            current_term, doc_ids, read_buffer_index = heapq.heappop(min_heap)
            read_buffers_to_advance.append(read_buffer_index)
            for doc_id in doc_ids:
                if doc_id not in doc_ids_with_current_term:
                    doc_ids_with_current_term.append(doc_id)
        doc_ids_with_current_term.sort()

        # adjust read buffer pointers
        for read_buffer_index in read_buffers_to_advance:
            read_buffer = read_buffers[read_buffer_index]
            if read_buffer:
                term, doc_ids = read_buffer.pop(0)
                heapq.heappush(min_heap, (term, doc_ids, read_buffer_index))
            elif len(read_buffer) == 0:
                # read next 100 lines from corresponding block file
                block_file = block_files[read_buffer_index]
                for _ in range(READ_BUFFER_LINES_PER_FILE):
                    line = block_file.readline()
                    if not line:
                        break
                    term, postings_str = line.strip().split(":")
                    postings_list = list(map(int, postings_str.split(",")))
                    read_buffer.append((term, postings_list))
                if read_buffer:
                    term, doc_ids = read_buffer.pop(0)
                    heapq.heappush(min_heap, (term, doc_ids, read_buffer_index))
                else:
                    continue # No more lines to read from this block file

        write_buffer.append(f"{current_term}:{','.join(str(doc_id) for doc_id in doc_ids_with_current_term)}\n")
        if len(write_buffer) >= WRITE_BUFFER_LINES:
            final_index_file.writelines(write_buffer)
            write_buffer.clear()

    if write_buffer:
        final_index_file.writelines(write_buffer) # Flush remaining buffer to disk


# ---------------------------------------------------------
# 9) WRITE BUFFER MANAGEMENT
# ---------------------------------------------------------
# - Append merged term-line to write buffer
# - If write buffer reaches WRITE_BUFFER_LINES:
# flush (append) to final_index.txt
# - After merge loop ends:
# flush remaining write buffer
# ---------------------------------------------------------
# --> add your Python code here
# ---------------------------------------------------------
# 10) CLEANUP
# ---------------------------------------------------------
# - Close all open block files
# - Ensure final_index.txt is properly written
# ---------------------------------------------------------
for block_file in block_files:
    block_file.close()

# compare final_index.txt with comparison_final_index.txt to verify correctness
with open("final_index.txt", "r") as final_index, open("comparison_final_index.txt", "r") as comparison_index:
    final_index_lines = final_index.readlines()
    comparison_index_lines = comparison_index.readlines()
    if final_index_lines == comparison_index_lines:
        print("Final index matches the expected output!")
    else:
        print("Final index does NOT match the expected output.")
