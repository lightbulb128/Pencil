import pickle, hashlib
import os

def save_preprocess_file(key:str, obj):
  hashed = "tk" + hashlib.sha256(key.encode()).hexdigest()[:8]
  print(f"Saved {key} -> {hashed}")
  filename = f"preprocess/{hashed}.pickle"
  file = open(filename, "wb")
  pickle.dump((key, obj), file)
  file.close()

def load_preprocess_file(key:str):
  hashed = "tk" + hashlib.sha256(key.encode()).hexdigest()[:8]
  filename = f"preprocess/{hashed}.pickle"
  if os.path.exists(filename):
    f = open(filename, "rb")
    loaded_key, obj = pickle.load(f)
    f.close()
    assert(loaded_key == key)
    return obj
  else:
    return None

def ceil_div(a, b):
  if a%b==0: return a//b
  return a//b+1