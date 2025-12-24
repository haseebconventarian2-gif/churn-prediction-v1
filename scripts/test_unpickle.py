import pickle
import traceback
import os

p = os.path.join('artifacts', 'preprocessing.pkl')
print('preprocessor path:', p)
print('exists:', os.path.exists(p))
try:
    with open(p, 'rb') as f:
        obj = pickle.load(f)
    print('unpickle succeeded, type:', type(obj))
except Exception:
    print('unpickle failed; traceback:')
    traceback.print_exc()
