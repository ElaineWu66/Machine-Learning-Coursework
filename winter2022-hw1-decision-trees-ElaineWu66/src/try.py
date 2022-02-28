import numpy as np
import math
'''
a = [[1,2,1],[1,1,1],[0,1,0], [0,0,0]]
y = [1,2,3,4]
a = np.array(a)
y = np.array(y)
print(a)
print(type(a))
b = y[np.where(a[:,1] == 1)]
print(type(b))
print(b)

list1 = [0,0,0,0,0]
list1 = np.array(list1)
list2 = []

for i in range(14):
    if i not in list1:
        list2.append(i)

print(np.random.choice(np.arange(0, 2), p=[0.5, 0.5]))
'''


list1 = [10,3,4,5,7,9]
largest_index = list1[0]
print("before: ",largest_index)
for i in list1:
    if (i>largest_index):
        largest_index = i
print("after: ",largest_index)
'''
list2.remove(3)
print(list1)
print(list2)
'''