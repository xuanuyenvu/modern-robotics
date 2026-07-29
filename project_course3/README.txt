FOLDER CONTENTS
-----------------
code/UR5_parameter.py       - UR5 kinematic/dynamic parameters (provided with
                            the assignment)
code/IKinBodyIterates.py    - Implementation of the dynamics functions(Forward Dynamics, Mass Matrix, etc.) 
                            following the same algorithms as the Modern Robotics library. 
code/simulate.py            - main script, run with: python3 simulate.py
simulation1.csv             - results for scenario 1 (fall from homeposition, 3s, 301 rows)
simulation2.csv             - results for scenario 2 (joint 2 = -1 rad,5s, 501 rows)

METHOD
-------
- Uses Forward Dynamics (recursive Newton-Euler algorithm) with zero torque at every joint to compute joint 
accelerations theta_ddot.
- Integrates over time using 4th-order Runge-Kutta (RK4), at 100 integration steps per second (meets the 
assignment's minimum requirement).
- Energy conservation was verified: total energy (kinetic + potential) stays essentially constant throughout 
both simulations (< 0.1% deviation), confirming the physics is correct (i.e., no artificial energy is being 
added by integration error, as would happen with simple Euler integration).

Scenario 1: E(t=0)=14.689 J -> E(t=3s)=14.689 J
Scenario 2: E(t=0)=64.560 J -> E(t=5s)=64.516 J

Note: because this is a pure rigid-body forward-dynamics simulation with no collision detection, links may 
visually pass through each other in scenario 2 when joint angles swing through large ranges. This is an 
expected limitation of the model (not a bug) - energy conservation confirms the dynamics themselves are computed correctly.