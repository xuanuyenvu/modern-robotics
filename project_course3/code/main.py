import csv
import numpy as np

import IKinBodyIterates as IKinBodyIterates      
from UR5_parameter import Mlist, Glist, Slist, M01, M12, M23, M34, M45, M56, M67


G_GRAVITY = [0, 0, -9.81]          
FTIP = [0, 0, 0, 0, 0, 0]       
N_JOINTS = 6


def simulate_free_fall(thetalist0, total_time, samples_per_sec=100,
                        sub_steps=10):
    dt_record = 1.0 / samples_per_sec
    dt_integrate = dt_record / sub_steps       
    n_records = int(round(total_time * samples_per_sec)) + 1

    thetalist = np.array(thetalist0, dtype=float)
    dthetalist = np.zeros(N_JOINTS)            

    theta_history = np.zeros((n_records, N_JOINTS))
    dtheta_history = np.zeros((n_records, N_JOINTS))
    theta_history[0, :] = thetalist
    dtheta_history[0, :] = dthetalist

    taulist0 = np.zeros(N_JOINTS)  

    def accel(theta, dtheta):
        return IKinBodyIterates.ForwardDynamics(theta, dtheta, taulist0, G_GRAVITY, FTIP, Mlist, Glist, Slist)

    for k in range(1, n_records):
        for _ in range(sub_steps):
            h = dt_integrate
            k1_dth = dthetalist
            k1_ddth = accel(thetalist, dthetalist)

            k2_dth = dthetalist + 0.5 * h * k1_ddth
            k2_ddth = accel(thetalist + 0.5 * h * k1_dth,
                             dthetalist + 0.5 * h * k1_ddth)

            k3_dth = dthetalist + 0.5 * h * k2_ddth
            k3_ddth = accel(thetalist + 0.5 * h * k2_dth,
                             dthetalist + 0.5 * h * k2_ddth)

            k4_dth = dthetalist + h * k3_ddth
            k4_ddth = accel(thetalist + h * k3_dth,
                             dthetalist + h * k3_ddth)

            thetalist = thetalist + (h / 6.0) * (k1_dth + 2 * k2_dth
                                                  + 2 * k3_dth + k4_dth)
            dthetalist = dthetalist + (h / 6.0) * (k1_ddth + 2 * k2_ddth
                                                    + 2 * k3_ddth + k4_ddth)
        theta_history[k, :] = thetalist
        dtheta_history[k, :] = dthetalist

    return theta_history, dtheta_history


def save_csv(theta_history, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        for row in theta_history:
            writer.writerow(row)
    print(f"Saved {filename}  ({theta_history.shape[0]} rows, {theta_history.shape[1]} columns)")


def total_energy(thetalist, dthetalist):
    Mmat = IKinBodyIterates.MassMatrix(thetalist, Mlist, Glist, Slist)
    KE = 0.5 * dthetalist @ Mmat @ dthetalist
    PE = _potential_energy_fk(thetalist)
    return KE + PE


def _potential_energy_fk(thetalist):
    g = 9.81
    PE = 0.0
    Ms = [M01, M12, M23, M34, M45, M56]
    M_cumulative = np.eye(4)
    Slist_arr = np.array(Slist)
    for i in range(N_JOINTS):
        M_cumulative = M_cumulative @ Ms[i]
        T = IKinBodyIterates.FKinSpace(M_cumulative, Slist_arr[:, :i + 1], thetalist[:i + 1])
        mass = Glist[i][3, 3]
        z = T[2, 3]
        PE += mass * g * z
    return PE


if __name__ == "__main__":
    SUB_STEPS = 1  
    
    theta0_case1 = [0, 0, 0, 0, 0, 0]
    hist1, dhist1 = simulate_free_fall(theta0_case1, total_time=3.0,
                                        samples_per_sec=100,
                                        sub_steps=SUB_STEPS)
    save_csv(hist1, "../simulation1.csv")

    theta0_case2 = [0, -1, 0, 0, 0, 0]
    hist2, dhist2 = simulate_free_fall(theta0_case2, total_time=5.0,
                                        samples_per_sec=100,
                                        sub_steps=SUB_STEPS)
    save_csv(hist2, "../simulation2.csv")

    print("\nScenario 1")
    dt = 1.0 / 100
    for idx in [0, 50, 150, 300]:
        e = total_energy(hist1[idx], dhist1[idx])
        print(f"  t={idx*dt:5.2f}s   E={e: .6f} J")

    print("\nScenario 2")
    for idx in [0, 100, 300, 500]:
        e = total_energy(hist2[idx], dhist2[idx])
        print(f"  t={idx*dt:5.2f}s   E={e: .6f} J")