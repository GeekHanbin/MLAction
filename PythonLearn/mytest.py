import numpy as ny

random = ny.random.rand(4,4)
print(random)

matrandom = ny.mat(random)
print(matrandom)

print(matrandom*matrandom.I - ny.eye(4))