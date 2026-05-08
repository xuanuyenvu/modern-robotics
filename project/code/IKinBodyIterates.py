import numpy as np


def se3ToVec(se3mat):
    return np.r_[[se3mat[2][1], se3mat[0][2], se3mat[1][0]],
                 [se3mat[0][3], se3mat[1][3], se3mat[2][3]]]



def MatrixLog6(T):
    R, p = TransToRp(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                     [[0, 0, 0, 0]]]



def TransToRp(T):
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]



def MatrixLog3(R):
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)
    
    
    
def NearZero(z):
    return abs(z) < 1e-6



def VecToso3(omg):
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])



def TransInv(T):
    R, p = TransToRp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]



def FKinBody(M, Blist, thetalist):
    T = np.array(M)
    for i in range(len(thetalist)):
        T = np.dot(T, MatrixExp6(VecTose3(np.array(Blist)[:, i] \
                                          * thetalist[i])))
    return T



def MatrixExp6(se3mat):

    se3mat = np.array(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatrixExp3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat,omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]
    


def MatrixExp3(so3mat):
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)



def so3ToVec(so3mat):
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

    
        
def VecTose3(V):
    return np.r_[np.c_[VecToso3([V[0], V[1], V[2]]), [V[3], V[4], V[5]]],
                 np.zeros((1, 4))]



def AxisAng3(expc3):
    return (Normalize(expc3), np.linalg.norm(expc3))



def Normalize(V):
    return V / np.linalg.norm(V)



def JacobianBody(Blist, thetalist):
    Jb = np.array(Blist).copy().astype(float)
    T = np.eye(4)
    for i in range(len(thetalist) - 2, -1, -1):
        T = np.dot(T,MatrixExp6(VecTose3(np.array(Blist)[:, i + 1] \
                                         * -thetalist[i + 1])))
        Jb[:, i] = np.dot(Adjoint(T), np.array(Blist)[:, i])
    return Jb



def Adjoint(T):
    R, p = TransToRp(T)
    return np.r_[np.c_[R, np.zeros((3, 3))],
                 np.c_[np.dot(VecToso3(p), R), R]]
    


def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
# Initial guess
    thetalist = np.array(thetalist0).copy()

    # Store all iterations
    iterates = []
    iterates.append(thetalist.copy())

    # Iteration settings
    i = 0
    maxiterations = 20

    # Initial error twist
    Vb = se3ToVec(
        MatrixLog6(
            np.dot(
                TransInv(FKinBody(M, Blist, thetalist)),
                T
            )
        )
    )

    # Error magnitudes
    omega_err = np.linalg.norm(Vb[0:3])
    v_err = np.linalg.norm(Vb[3:6])

    # Check convergence
    err = omega_err > eomg or v_err > ev

    # Forward kinematics
    Tsb = FKinBody(M, Blist, thetalist)

    # Print iteration 0
    print(f"\nIteration {i}:")

    print("\nJoint Vector:")
    print(", ".join(f"{x:.3f}" for x in thetalist))

    print("\nSE(3) End-Effector Configuration:")
    print(np.round(Tsb, 3))

    print("\nError Twist Vb:")
    print(", ".join(f"{x:.3f}" for x in Vb))

    print(f"\nAngular Error Magnitude ||ω_b||: {omega_err:.6f}")
    print(f"Linear Error Magnitude ||v_b||: {v_err:.6f}")

    # Newton-Raphson loop
    while err and i < maxiterations:

        # Newton step
        thetalist = thetalist + np.dot(
            np.linalg.pinv(
                JacobianBody(Blist, thetalist)
            ),
            Vb
        )

        # Save iteration
        iterates.append(thetalist.copy())

        # Update iteration counter
        i += 1

        # Compute new error twist
        Vb = se3ToVec(
            MatrixLog6(
                np.dot(
                    TransInv(FKinBody(M, Blist, thetalist)),
                    T
                )
            )
        )

        # Compute FK
        Tsb = FKinBody(M, Blist, thetalist)

        # Compute errors
        omega_err = np.linalg.norm(Vb[0:3])
        v_err = np.linalg.norm(Vb[3:6])

        # Print iteration info
        print(f"\nIteration {i}:")

        print("\nJoint Vector:")
        print(", ".join(f"{x:.3f}" for x in thetalist))

        print("\nSE(3) End-Effector Configuration:")
        print(np.round(Tsb, 3))

        print("\nError Twist Vb:")
        print(", ".join(f"{x:.3f}" for x in Vb))

        print(f"\nAngular Error Magnitude ||ω_b||: {omega_err:.6f}")
        print(f"Linear Error Magnitude ||v_b||: {v_err:.6f}")

        # Check convergence
        err = omega_err > eomg or v_err > ev

    # Save all iterates to CSV
    np.savetxt(
        "../iterates.csv",
        np.array(iterates),
        delimiter=","
    )

    return (thetalist, not err)