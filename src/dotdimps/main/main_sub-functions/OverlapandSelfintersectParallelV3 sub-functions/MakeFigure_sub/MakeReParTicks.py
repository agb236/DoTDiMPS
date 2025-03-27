import numpy as np

def MakeReParTicks(RePar, n):
    min_val = RePar[0]
    max_val = RePar[-1]
    innerlable = np.unique(np.ceil(np.linspace(min_val, max_val, n) / 10) * 10)
    innerlable[0] = min_val
    innerlable[-1] = max_val
    inner = np.zeros(innerlable.shape)
    for i in range(len(inner)):
        inner[i] = np.argmax(RePar >= innerlable[i]) + 1 # +1 because of the 1-indexing in MATLAB (maybe wrong if uesed to index later)
    return innerlable, inner


#RePar1 = data = np.loadtxt("C:/Users/Kapta/Documents/Skole/DTU/6.semester/BP/Python code/Oversæt/Test txt/MakeReParTicks/RePar1.txt")
#n = 8
#MakeReParTicks(RePar1,n)