import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

# --- Configuration & Pathing ---
load_dotenv()

# This finds the absolute path of the script, goes up one level to 'gpt2_implementation',
# and up another level to the project root where 'data/' lives.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE_DIR = REPO_ROOT / "data" / "fineweb"

# Create the directory if it doesn't exist (parents=True handles the /data/ part too)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

remote_name = "sample-10BT"
shard_size = int(1e8)

# --- Tokenizer Setup ---
# GPT-2 uses the 'gpt2' encoding; tiktoken is significantly faster than transformers
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens."""
    tokens = [eot] # Start each doc with the <|endoftext|> delimiter
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    
    # uint16 supports up to 65535, which fits GPT-2's 50257 vocab size
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    """Saves the token array as a .npy file."""
    np.save(filename, tokens_np)

# --- Main Execution Block ---
# Essential for multiprocessing to work on Windows/macOS/latest Linux distros
if __name__ == '__main__':
    print(f"Target directory: {DATA_CACHE_DIR}")
    
    # 1. Download/Stream the dataset
    print("Loading FineWeb-Edu dataset...")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # 2. Setup Multiprocessing Pool
    # Using half of available CPUs is usually optimal to avoid IO bottlenecks
    nprocs = max(1, os.cpu_count() // 2)
    
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # Preallocate buffer to hold current shard (100M tokens)
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        
        # pool.imap handles the mapping of tokenize() over the dataset entries
        for tokens in pool.imap(tokenize, fw, chunksize=16):

            # Check if current shard has room for the new document
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            
            else:
                # Shard is full. The first shard (0) is usually saved as 'val'
                split = "val" if shard_index == 0 else "train"
                filename = DATA_CACHE_DIR / f"edufineweb_{split}_{shard_index:06d}.npy"
                
                # Fill the remaining gap in the current shard
                remainder = shard_size - token_count
                if progress_bar:
                    progress_bar.update(remainder)
                
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                
                # Write to disk
                write_datafile(str(filename), all_tokens_np)
                shard_index += 1
                progress_bar = None # Reset progress bar for next shard
                
                # Carry over the leftover tokens from this doc to the start of the next shard
                leftovers = tokens[remainder:]
                all_tokens_np[0:len(leftovers)] = leftovers
                token_count = len(leftovers)

        # 3. Handle the final partial shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = DATA_CACHE_DIR / f"edufineweb_{split}_{shard_index:06d}.npy"
            write_datafile(str(filename), all_tokens_np[:token_count])
            print(f"\nProcessing complete. Final shard saved to {filename}")