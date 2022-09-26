# score-range-aided-slam

Code to solve a second-order cone program as a convex relaxation of the range-aided SLAM problem

## Dependencies

### PyFactorGraph

[Link to the repo](https://github.com/MarineRoboticsGroup/PyFactorGraph)

This holds all of the measurements/variables to define our RA-SLAM problem.
This is a custom library developed in the Marine Robotics Group at MIT to 
interface with a broader range of SLAM file types. You can install directly 
from source via `pip install .` inside the home directory

### Drake Dependency (read carefully)

Drake is a wonderful piece of software... once it's installed. Installing the
python bindings for Drake can be a hassle. We will have to build Drake from
source to use the `Gurobi` solver, which is the real reason we are using Drake
(it gives us a great interface to the solver). Make sure that you have Gurobi
properly set up before doing any of this.

Useful links

- <https://drake.mit.edu/from_source.html#mandatory-platform-specific-instructions>
- <https://drake.mit.edu/bazel.html#proprietary-solvers>
- <https://drake.mit.edu/pydrake/pydrake.solvers.gurobi.html>

### GTSAM (optional)

We used GTSAM to refine our initial estimates provided by SCORE. We recommend 
installing via `pip install gtsam==4.1.0`.

## Usage

Feel free to look inside our `/examples` directory. You can also directly call `python3 score/solve_score.py`.

## Paper

Extended version of our ICRA 2023 Submission: [SCORE Extended Version](https://github.com/alanpapalia/score2022extended/blob/master/score2023_extended.pdf)
