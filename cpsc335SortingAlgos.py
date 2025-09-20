from typing import List, Dict, Tuple
import time
import random

def bubble_sort(arr):   #Define a bubble sort function that takes a list as an input

    n = len(arr)

    for i in range(n):
        swapped = False

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

def counting_sort(arr):   #will only work with non-negative integers

    if not arr:
        return []
    
    max_val = max(arr)
    min_val = min(arr)
    #offset = -min_val if min_val < 0 else 0
    k = max_val - min_val + 1

    count = [0] * k
    for num in arr:
        count[num - min_val] += 1

    output = []
    for i, freq in enumerate(count):
        value = i + min_val
        output.extend([value] * freq)
        
    return output

def heap_sort(arr):
#Heap sort would be in place as worst case effiecency with 0(n log n)
    
    def sift_down(a, start, end):
        root = start
        while (left := 2 * root + 1) <= end:
            right = left + 1
            largest = root
            if a[left] > a[largest]:
                largest = left
            if right <= end and a[right] > a[largest]:
                largest = right
            if largest == root:
                break
            a[root], a[largest] = a[largest], a[root]
            root = largest
    def build_max_heap(a):
        n = len(a)
        for i in range(n // 2 - 1, -1, -1):
            sift_down(a, i, n - 1)
    
    a = arr
    n = len(a)
    build_max_heap(a)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end - 1)
    return a

def insertion_sort(arr):
    n = len(arr)

    for  i in range(1, n):   #starts with second element in given array
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:   #shifts elements that are greater than the key value
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key
    return arr

#Merge Sort
def merge_sort(arr):

    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    return merge(left_half, right_half)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort(arr):

    def partition(low, high):
        pivot = arr[(low + high) // 2]
        i = low
        j = high
        while i <= j:
            while arr[i] < pivot:
                i += 1
            while arr[j] > pivot:
                j -= 1
            if i <= j:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
                j -= 1
        return i, j
    
    def sort(low, high):
        if low < high:
            i, j = partition(low, high)
            sort(low, j)
            sort(i, high)
    sort(0, len(arr) - 1)
    return arr

# Core: stable counting sort by one digit (base would be 10 by default)

def _counting_sort_by_digit(a: List[int], exp: int, base: int = 10) -> None:
    n = len(a)
    output = [0] * n
    count = [0] * base

    #1 Count occurances of ths digit among all #'s
    for i in range(n):
        digit = (a[i] // exp) % base
        count[digit] += 1

    #2 Prefix Sums: transform count into ending positions
    for d in range(1, base):
        count[d] += count[d - 1]

    #3 Stable write: traverse input backwards to preserve order of equal digits
    for i in range(n - 1, -1, -1):
        digit = (a[i] // exp) % base
        pos = count[digit] - 1
        output[pos] = a[i]
        count[digit] -= 1

    #4 Copt back: make this pass's result become the new array for the next digit
    for i in range(n):
        a[i] = output[i]



#Define Radix Sort LSD for Non Negative Integers

def radix_sort_lsd_nonneg(a: List[int], base: int = 10) -> List[int]:
    if not a:
        return a

    start = time.perf_counter() #Starting high resolution timer for benchmarking

    max_val = max(a)
    exp = 1
    
    while max_val // exp > 0:
        _counting_sort_by_digit(a, exp, base)
        exp *= base

    end = time.perf_counter()
    print(f"[Radix] nonneg sort in {end - start:.6f} sec base={base}")  #{base={}}
    return a

#Handling negative values
def radix_sort_lsd(a: List[int], base: int = 10) -> List[int]:
    if not a:
        return a

    neg = [-x for x in a if x < 0]
    pos = [x for x in a if x >= 0]

    #Sort both subsets
    if neg:
        radix_sort_lsd_nonneg(neg, base)
    if pos:
        radix_sort_lsd_nonneg(pos, base)

    neg_sorted = [-x for x in reversed(neg)]

    out = neg_sorted + pos
    return out

#Real Worlds scenario - Sorting Order Records
def sort_orders_by_id(orders: List[Dict], key: str = "order_id", base: int = 10) -> List[Dict]:
    if not orders:
        return orders

    keys = []
    indx = []

    for i, rec in enumerate(orders):
        keys.append(int(rec[key]))
        indx.append(i)

    def stable_pass_with_companion(keys: List[int], comp: List[int], exp: int, base: int) -> None:
        n = len(keys)
        out_keys = [0] * n
        out_comp = [0] * n
        count = [0] * base
        for i in range(n):
            d = (keys[i] // exp) % base
            count[d] += 1
        for d in range(1, base):
            count[d] += count[d - 1]
        for i in range(n - 1, -1, -1):
            d = (keys[i] // exp) % base
            pos = count[d] - 1
            out_keys[pos] = keys[i]
            out_comp[pos] = comp[i]
            count[d] -= 1
        for i in range(n):
            keys[i] = out_keys[i]
            comp[i] = out_comp[i]

    #Run LSD passes
    max_key = max(keys)
    exp = 1
    start = time.perf_counter()
    while max_key // exp > 0:
        stable_pass_with_companion(keys, indx, exp, base)
        exp *= base
    end = time.perf_counter()
    print(f"[Radix] order sort in {end - start:.6f} sec for {len(orders)} records")

    sorted_orders = [orders[i] for i in indx]
    return sorted_orders

def bucket_sort(arr: List[float]) -> List[float]:

    n = len(arr)
    if n == 0:
        return arr

    start = time.perf_counter()

    buckets = [[] for _ in range (n)]

    for x in arr:
        idx = int(n * x)
        buckets[idx].append(x)

    for i in range(n):
        buckets[i].sort()

    result = []
    for b in buckets: 
        result.extend(b)

    end = time.perf_counter()
    print(f"[Bucket] sort of {n} elements in {end-start:.6f} sec")

    return result

def partition(arr: List[int], low: int, high: int) -> int:

    pivot = arr[high]
    i = low
    for j in range(low, high):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i

def quick_select(arr: List[int], low: int, high: int, k: int) -> int:
    #Recursive Quick Select

    if low <= high:
        pi = partition(arr, low, high)
        if pi == k:
            return arr[pi]
        elif pi > k:
            return quick_select(arr, low, pi-1, k)
        else:
            return quick_select(arr, pi+1, high, k)

def timed_quick_select(arr: List[int], k: int) -> int:

    start = time.perf_counter()
    result = quick_select(arr, 0, len(arr)-1, k)
    end = time.perf_counter()
    print(f"[Quick Select] found k={k} in {end-start:.6f} sec")
    return result
