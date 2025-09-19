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
