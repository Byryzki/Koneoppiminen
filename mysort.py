from operator import indexOf
import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')

for idx, arr_val in enumerate(arr):
    arr[idx] = int(arr_val)

# Print
print(f'Before sorting {arr}')

# My sorting (e.g. bubble sort)

def pyrysort(oglist):
    for i, nro in enumerate(arr):
        if i != len(arr)-1:
            if nro <= arr[i + 1]:
                continue
            else:
                x = arr[i]
                arr[i] = arr[i+1]
                arr[i+1] = x
        else:
            return arr

# Print
for j in arr:
    pyrysort(arr)

print(f'After sorting {arr}')