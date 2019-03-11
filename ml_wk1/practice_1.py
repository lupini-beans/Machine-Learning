import numpy as np

print('___Basic Data Types___')
print('Numbers:')
x = 3
print(x ** 2)
t = True
f = False
print(t != f)

print()

print('Strings:')
hello = 'hello'
print(len(hello))
hw = hello + ' world'
print(hw)

print()
print('___Containers___')
print('Lists:')
xs = [3, 1, 2]
print(xs, xs[2])
print(xs[-1])
xs[2] = 'foo'
print(xs)

print()

print('Slicing:')
nums = list(range(5))
print(nums)
print(nums[2:4])
print(nums[2:])
print(nums[:-1])

print()

print('Loops:')
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

print('...')
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))

print()

print('Dictionaries:')
d = {'cat': 'cute', 'dog': 'furry'}
print(d['cat'])

print()

print('Tuples:')
d = {(x, x + 1): x for x in range(10)}
t = (5,6)
print(type(t))
print(d[t])
print(d[(1,2)])

print()

print('Functions:')
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

print('___Numpy___')
print('Arrays')
b = np.array([[1,2,3],[4,5,6]])
print(b.shape)
print(b[0,0], b[0,1], b[1,0])

print('...')
a = np.zeros((2,2))
print(a)

b = np.ones((1,2))
print(b)

c = np.full((2,2), 7)
print(c)

d = np.eye(2)
print(d)

e = np.random.random((2,2))
print(e)

print()

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
b = a[:2, 1:3]
print(a[0, 1])
b[0, 0] = 77
print(a[0, 1])

print()
print('Array math:')
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
print(x + y)
print(np.add(x,y))

print(x * y)
print(np.multiply(x,y))

v = np.array([9,10])
print(x.dot(v))
print(np.dot(x,v))

print()
print('Broadcasting')
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4,1))
print(vv)
y = x + vv
print(y)

