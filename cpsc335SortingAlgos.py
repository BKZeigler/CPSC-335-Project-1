from typing import List, Dict, Tuple
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Button

def bubble_sort(arr):   #Define a bubble sort function that takes a list as an input
    """Sorts a list using the Bubble Sort algorithm"""
    n = len(arr)

    #iterate the nested loop enough times for it to sort the smallest element, if needed
    for i in range(n):
        #flag that determines if further sorting is necessary
        swapped = False

		#shifts the greatest unsorted element to its proper place on the list
		#ex. the 2nd greatest element is shifted to the 2nd-to-last index
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        #no need to sort further
        if not swapped:
            break
    
    return arr

def counting_sort(arr):   #will only work with non-negative integers
    """Sorts a list of nonnegative integers using the Counting Sort algorithm"""
    #checks for empty list
    if not arr:
        return []
    
    #find the number of possible values for the elements in given list
    max_val = max(arr)
    min_val = min(arr)
    #offset = -min_val if min_val < 0 else 0
    k = max_val - min_val + 1

    #list of integers representing the frequency of each possible value
    #index 0 represents the minimum value in the list, index 1 represents minimum value + 1, and so on
    count = [0] * k
    for num in arr:
        count[num - min_val] += 1

    #sorted list created using the indexes and elements of count[]
    output = []
    for i, freq in enumerate(count):
        value = i + min_val
        output.extend([value] * freq)
        
    return output

def heap_sort(arr):
#Heap sort would be in place as worst case effiecency with 0(n log n)
    """Sorts a list using the Heap Sort algorithm"""
    def sift_down(a, start, end):
        root = start
        while (left := 2 * root + 1) <= end:
            right = left + 1
            largest = root
            #print("left:", left, "right:", right, "largest:", largest, "root:", root)
            if a[left] > a[largest]:
                largest = left
            if right <= end and a[right] > a[largest]:
                largest = right
            if largest == root:
                break
            a[root], a[largest] = a[largest], a[root]
            #print(a)
            root = largest
    
    def build_max_heap(a):
        n = len(a)
        for i in range(n // 2 - 1, -1, -1):
            sift_down(a, i, n - 1)
            #print("sift_down() at index", i, ":", a)
    
    a = arr
    n = len(a)
    build_max_heap(a)
    #print("After build_max_heap():", a)
    for end in range(n - 1, 0, -1):
		#swap 
        a[0], a[end] = a[end], a[0]
        
        sift_down(a, 0, end - 1)
    return a

def insertion_sort(arr):
    """Sorts a list using the Insertion Sort algorithm"""
    n = len(arr)

	#iterate insertion sort with each element as the "key"
    for  i in range(1, n):   #starts with second element in given array
        key = arr[i]
        j = i - 1

		#shifts forward all elements greater than the key value
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

		#reassign stored key value to proper place in list
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    """Sorts a list using the Merge Sort algorithm"""
	#checks for empty or single-item list; end condition
    if len(arr) <= 1:
        return arr
    
    #recursively cuts list into halves
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])

    return merge(left_half, right_half)

def merge(left, right):
	"""Returns """
	result = []
	i = j = 0		#iterators for each half

	#appends the smallest elements from the left and right to result
	while i < len(left) and j < len(right):
		if left[i] <= right[j]:
			result.append(left[i])
			i += 1
		else:
			result.append(right[j])
			j += 1

	#tacks on remaining (big) elements to result
	result.extend(left[i:])
	result.extend(right[j:])
	return result


def quick_sort(arr):
    """Sorts a list using the Quick Sort algorithm"""

    def partition(low, high):
        pivot = arr[(low + high) // 2]		#pivot value set to element in (effectively) middle index
        i = low
        j = high
        while i <= j:
			#iterate forwards to find a value >= pivot
            while arr[i] < pivot:
                i += 1
            
            #iterate backwards to find a value <= pivot
            while arr[j] > pivot:
                j -= 1
            
            #swaps the elements at indexes i & j such that arr[i] < pivot < arr[j]
            if i <= j:
                print(arr[i], pivot, arr[j])
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
    """Sorts an array using the Counting Sort by Digit algorithm"""
    n = len(a)
    output = [0] * n
    count = [0] * base

    #1 Count occurances of certain digit over all #'s
    for i in range(n):
        digit = (a[i] // exp) % base
        count[digit] += 1

    #2 Prefix sums: transform count into ending positions
    for d in range(1, base):
        count[d] += count[d - 1]

    #3 Stable write: traverse input backwards to preserve order of equal digits
    for i in range(n - 1, -1, -1):
        digit = (a[i] // exp) % base
        pos = count[digit] - 1
        output[pos] = a[i]
        count[digit] -= 1

    #4 Copy back: make this pass's result become the new array for the next digit
    for i in range(n):
        a[i] = output[i]



#Define Radix Sort LSD for Non Negative Integers

def radix_sort_lsd_nonneg(a: List[int], base: int = 10) -> List[int]:
    if not a:
        return a

    #start = time.perf_counter() #Starting high resolution timer for benchmarking

    max_val = max(a)
    exp = 1
    
    while max_val // exp > 0:
        _counting_sort_by_digit(a, exp, base)
        exp *= base

    #end = time.perf_counter()
    #print(f"[Radix] nonneg sort in {end - start:.6f} sec base={base}")  #{base={}}
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
    #start = time.perf_counter()
    while max_key // exp > 0:
        stable_pass_with_companion(keys, indx, exp, base)
        exp *= base
    #end = time.perf_counter()
    #print(f"[Radix] order sort in {end - start:.6f} sec for {len(orders)} records")

    sorted_orders = [orders[i] for i in indx]
    return sorted_orders

def bucket_sort(arr: List[float]) -> List[float]:
    """Sorts an array using the Bucket Sort algorithm"""
    n = len(arr)
    if n == 0:
        return arr

    #start = time.perf_counter()

    buckets = [[] for _ in range (n)]

    max_val = max(arr) # test
    for x in arr:
        idx = int((x / (max_val + 1)) * n)
        #idx = int(n * x)
        buckets[idx].append(x)

    for i in range(n):
        buckets[i].sort()

    result = []
    for b in buckets: 
        result.extend(b)

    #end = time.perf_counter()
    #print(f"[Bucket] sort of {n} elements in {end-start:.6f} sec")

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

def pre_quick_select(arr):
        if arr:
            k = len(arr) // 2
        else:
            return arr
        return quick_select(arr, 0, len(arr) - 1, k)

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


#Plot Data
fig, axes = plt.subplots()
algos_names = ["Bucket", "Quick Select", "Bubble", "Counting", "Heap", "Insertion", "Merge", "Radix"] #"Bubble", "Counting", "Heap", "Insertion", "Merge", "Quick", "Radix",
algos_sort = [bucket_sort, pre_quick_select, bubble_sort, counting_sort, heap_sort, insertion_sort, merge_sort, radix_sort_lsd_nonneg]
algos_sort_negative = [bucket_sort, pre_quick_select, bubble_sort, counting_sort, heap_sort, insertion_sort, merge_sort, radix_sort_lsd]
algos_times = []
plt.xticks(rotation=45, ha="right")  #Makes the x labels at an angle so no collision
plt.subplots_adjust(bottom=0.4)
#numbers = [random.randint(-100,100) for a in range(10)]  #10 random numbers from -100-100

choice = input("Enter numbers? (Y/N)")
if choice == 'Y':
    numbers = list(map(int, input("Enter comma seperated numbers:").split(","))) #takes out commas
else:
    numbers = [random.randint(-100,100) for a in range(10)]

axes.set_xlabel("Sorting Algorithims") #Titles
axes.set_ylabel("Execution Time")
axes.set_title("Sorting Algorithim Performance")

is_negative = False
for num in numbers: #for every number in numbers list
    if num < 0: #if a number is negative
        is_negative = True #show there is a negative in the list
        break #exit

if is_negative == False: #No negative values
    for algo in algos_sort: #use radix sort for no negatives
        numbers_copy = numbers.copy()
        start = time.time()
        algo(numbers_copy)
        end = time.time()
        algos_times.append(end - start)
else: #There are negative values
    for algo in algos_sort_negative: #use radix sort for negatives
        numbers_copy = numbers.copy()
        start = time.time()
        algo(numbers_copy)
        end = time.time()
        algos_times.append(end - start)

print("Algorithm Times in Microseconds:") #printing algorithm name next to its respective time
for name, t in zip(algos_names, algos_times):   #combine the 2 lists to be used as a dictionary
    print(f"{name}: {t * 1e6:.2f} microseconds")

bars = axes.bar(algos_names, algos_times) #sets x to names and y to times

def update(frame): #set bar to this height every update (frame)
    for bar, height in zip(bars, np.array(algos_times) * frame / 20):
        bar.set_height(height)
    return bars

bar_animation = animation.FuncAnimation(fig, update, frames=20, interval=100, repeat=False) #creates animation

#Start, Reset, Pause Functionality
def start(event):
    if bar_animation and bar_animation.event_source:
        bar_animation.event_source.start() #continues the animation

def pause(event):
    if bar_animation and bar_animation.event_source:
        bar_animation.event_source.stop() #stop the current animation

def reset(event):
    global bar_animation
    for bar in bars:
        bar.set_height(0) #reset the bars
    fig.canvas.draw_idle()
    if bar_animation:
        bar_animation.frame_seq = bar_animation.new_frame_seq() # reset to frame 0
        if bar_animation.event_source:
            bar_animation.event_source.stop()
        else:
            bar_animation = animation.FuncAnimation(fig, update, frames=20, interval=100, repeat=False)


axes_start = plt.axes([0.3, 0.05, 0.1, 0.075]) #Positioning buttons
axes_pause = plt.axes([0.45, 0.05, 0.1, 0.075])
axes_reset = plt.axes([0.6, 0.05, 0.1, 0.075])
btn_start = Button(axes_start, "Start") #Place buttons and name them
btn_pause = Button(axes_pause, "Pause")
btn_reset = Button(axes_reset, "Reset")
btn_start.on_clicked(start) #when button is clicked, call respective function
btn_pause.on_clicked(pause)
btn_reset.on_clicked(reset)


plt.show()
