import numpy as np
from IKinBodyIterates import IKinBodyIterates

L1 = 0.425
L2 = 0.392
W1 = 0.109
W2 = 0.082
H1 = 0.089
H2 = 0.095

M = np.array([
    [-1, 0, 0, L1 + L2],
    [0, 0, 1, W1 + W2],
    [0, 1, 0, H1 - H2],
    [0, 0, 0, 1]
])

Blist = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0,-1, 0],
    [0, 1, 1, 1, 0, 1],
    [W1+W2, H2, H2, H2, -W2, 0],
    [0, -(L1+L2), -L2, 0, 0, 0],
    [L1+L2, 0, 0, 0, 0, 0]
])

Tsd = np.array([
    [0, 1, 0, -0.5],
    [0, 0,-1, 0.1],
    [-1,0, 0, 0.1],
    [0,0,0,1]
])

thetalist0 = np.array([2.3, -0.8, 1.3, -0.5, -0.3, -1.3])

eomg = 0.001
ev = 0.0001

solution, success = IKinBodyIterates(
    Blist,
    M,
    Tsd,
    thetalist0,
    eomg,
    ev
)

print("Success:", success)
print("Solution (radians):")
print(solution)