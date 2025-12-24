p='artifacts/preprocessing.pkl'
with open(p,'rb') as f:
    data=f.read()
print('size', len(data))
print('contains numpy._core?', b'numpy._core' in data)
idx = data.find(b'numpy._core')
print('index', idx)
if idx!=-1:
    start = max(0, idx-60)
    print(data[start:idx+60])
else:
    # print some common module strings around pickle
    for name in [b'numpy.core', b'sklearn', b'sklearn.impute']:
        print(name, name in data)
