import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import List, Dict, Tuple, MutableMapping, Union
from score.utils.matrix_utils import get_random_transformation_matrix
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.measurements import (
    FGRangeMeasurement,
    POSE_MEASUREMENT_TYPES,
    PoseMeasurement2D,
    PoseMeasurement3D,
)
from py_factor_graph.priors import LandmarkPrior2D, LandmarkPrior3D
from py_factor_graph.utils.solver_utils import (
    SolverResults,
    VariableValues,
    save_to_tum,
)
from py_factor_graph.utils.plot_utils import visualize_solution
from py_factor_graph.utils.matrix_utils import round_to_special_orthogonal
from attrs import define, field, validators

acceptable_relaxations = ["SOCP", "QCQP"]

# import logging

# logger = logging.getLogger(__name__)


def is_dimension(instance, attribute, value) -> None:
    """
    Return validator for dimension.

    Args:
        value (int): value to validate

    Returns:
        None
    """
    if not isinstance(value, int):
        raise ValueError(f"{value} is not an int")
    if not value in [2, 3]:
        raise ValueError(f"Value {value} is not 2 or 3")


@define
class VariableCollection:
    dim: int = field(validator=is_dimension)
    pose_vars: Dict[str, gp.MVar] = field(factory=dict)
    landmark_vars: Dict[str, gp.MVar] = field(factory=dict)
    distance_vars: MutableMapping[Tuple[str, str], Union[gp.MVar, gp.Var]] = field(
        factory=dict
    )

    def _check_is_new_variable(self, var_name: Union[str, Tuple[str, str]]):
        if isinstance(var_name, tuple):
            if var_name in self.distance_vars:
                raise ValueError(
                    f"Variable name {var_name} already exists in distance_vars"
                )
        elif isinstance(var_name, str):
            if var_name in self.pose_vars:
                raise ValueError(
                    f"Variable name {var_name} already exists in pose_vars"
                )
            if var_name in self.landmark_vars:
                raise ValueError(
                    f"Variable name {var_name} already exists in landmark_vars"
                )
        else:
            raise ValueError(
                f"Variable name {var_name} is not a valid type: {type(var_name)}"
            )

    def add_pose_variable(self, pose_var: gp.MVar, name: str):
        self._check_is_new_variable(name)
        self.pose_vars[name] = pose_var

    def add_landmark_variable(self, landmark_var: gp.MVar, name: str):
        self._check_is_new_variable(name)
        self.landmark_vars[name] = landmark_var

    def add_distance_variable(self, distance_var: gp.MVar, dist_key: Tuple[str, str]):
        self._check_is_new_variable(dist_key)
        self.distance_vars[dist_key] = distance_var

    def set_distance_variables(self, dist_vars: gp.tupledict):
        assert (
            self.distance_vars == {}
        ), f"Trying to set distance variables when it is non-empty, this is not the intended usage: {self.distance_vars}"
        self.distance_vars = dist_vars

    def get_pose_var(self, pose_name: str) -> gp.MVar:
        return self.pose_vars[pose_name]

    def get_translation_var(self, var_name: str) -> gp.MVar:
        if var_name in self.pose_vars:
            return self.pose_vars[var_name][:, -1]
        elif var_name in self.landmark_vars:
            return self.landmark_vars[var_name]
        else:
            raise ValueError(f"Variable name {var_name} not found")

    def get_distance_var(self, dist_key: Tuple[str, str]) -> gp.MVar:
        return self.distance_vars[dist_key]

    def get_variable_values(self) -> VariableValues:
        def _clean_pose_est(pose_est: np.ndarray) -> np.ndarray:
            # each pose needs to be homogenized and rotation matrix needs to be
            # orthogonalized
            rot_est = pose_est[: self.dim, : self.dim]
            rounded_rot = round_to_special_orthogonal(rot_est)

            # make a new (homogeneous) pose estimate with the rounded rotation
            new_pose_est = np.eye(self.dim + 1)
            new_pose_est[: self.dim, : self.dim] = rounded_rot
            new_pose_est[: self.dim, -1] = pose_est[: self.dim, -1]
            return new_pose_est

        def _clean_dist_est(dist_est: Union[float, np.ndarray]) -> np.ndarray:
            if isinstance(dist_est, float):
                return np.array([dist_est])
            else:
                return dist_est

        pose_vals = {k: _clean_pose_est(v.X) for k, v in self.pose_vars.items()}
        landmark_vals = {k: v.X for k, v in self.landmark_vars.items()}
        dist_vals = {k: _clean_dist_est(v.X) for k, v in self.distance_vars.items()}
        return VariableValues(self.dim, pose_vals, landmark_vals, dist_vals)


def _check_valid_relaxation(relaxation: str):
    if relaxation not in acceptable_relaxations:
        raise ValueError(
            f"Relaxation {relaxation} is not supported. "
            f"Acceptable relaxations are {acceptable_relaxations}"
        )


def vec_norm_sq(vec: Union[gp.MLinExpr, gp.MVar]) -> gp.MQuadExpr:
    """
    Returns the squared L2 norm of a vector.

    Args:
        vec (Union[gp.MLinExpr, gp.Mvar]): The vector to get the norm of.

    Returns:
        gp.MQuadExpr: The norm of the vector.
    """
    return gp.quicksum(vec * vec)


def mat_frob_norm_sq(mat: Union[gp.MLinExpr, gp.MVar]) -> gp.MQuadExpr:
    """
    Returns the squared Frobenius norm of a matrix.

    Args:
        mat (Union[gp.MLinExpr, gp.Mvar]): The matrix to get the norm of.

    Returns:
        gp.MQuadExpr: The norm of the matrix.
    """
    return gp.quicksum(gp.quicksum((mat * mat)))


def initialize_model(
    variables: VariableCollection,
    model: gp.Model,
    fg: FactorGraphData,
    relaxation_type: str,
):
    _check_valid_relaxation(relaxation_type)
    add_all_variables(variables, model, fg, relaxation_type)
    first_pose = fg.pose_variables[0][0]
    pose_var = variables.get_pose_var(first_pose.name)
    pin_pose(pose_var, model)
    add_distance_constraints(variables, model, relaxation_type)
    cost = get_full_cost_objective(variables, fg, relaxation_type)
    model.setObjective(cost)


def _extract_solver_results(
    model: gp.Model, vars: VariableCollection, data: FactorGraphData
) -> SolverResults:
    solver_vals = vars.get_variable_values()
    solve_time = model.Runtime
    solved = model.status == GRB.Status.OPTIMAL
    pose_chain_names = data.get_pose_chain_names()
    solve_results = SolverResults(
        variables=solver_vals,
        total_time=solve_time,
        solved=solved,
        pose_chain_names=pose_chain_names,
    )
    return solve_results


def solve_problem(data: FactorGraphData, relaxation_type: str) -> SolverResults:
    variables = VariableCollection(data.dimension)
    model = gp.Model()
    initialize_model(variables, model, data, relaxation_type)
    model.optimize()
    return _extract_solver_results(model, variables, data)


def solve_problem_with_intermediate_iterates(
    data: FactorGraphData, relaxation_type: str
) -> List[VariableValues]:
    iterates = []
    model_vars = VariableCollection(dim=data.dimension)
    model = gp.Model()
    initialize_model(model_vars, model, data, relaxation_type)
    curr_iter = 0
    while not finished_solving:
        model.Params.BarIterLimit = curr_iter
        model.optimize()

        # check if solver done
        if model.status != GRB.Status.ITERATION_LIMIT:
            finished_solving = True

        curr_solver_results = _extract_solver_results(model, model_vars, data)
        iterates.append(curr_solver_results)

        curr_iter += 1

    return iterates


##### set up variables #####


def add_all_variables(
    variables: VariableCollection,
    mod: gp.Model,
    fg: FactorGraphData,
    relaxation_type: str,
) -> None:
    _check_valid_relaxation(relaxation_type)
    add_pose_variables(variables, mod, fg)
    add_landmark_variables(variables, mod, fg)
    add_distance_variables(variables, mod, fg, relaxation_type)


def add_pose_variables(
    vars_collection: VariableCollection, mod: gp.Model, fg: FactorGraphData
):
    dim = fg.dimension
    for pose_chain in fg.pose_variables:
        for pose in pose_chain:
            pose_name = pose.name
            pose_var = mod.addMVar(
                shape=(dim, dim + 1),
                lb=-gp.GRB.INFINITY,
                ub=gp.GRB.INFINITY,
                name=pose_name,
            )
            vars_collection.add_pose_variable(pose_var, pose_name)


def add_landmark_variables(
    vars_collection: VariableCollection, mod: gp.Model, fg: FactorGraphData
):
    dim = fg.dimension
    for landmark_var in fg.landmark_variables:
        landmark_name = landmark_var.name
        landmark_var = mod.addMVar(
            shape=(dim,), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=landmark_name
        )
        vars_collection.add_landmark_variable(landmark_var, landmark_name)


def add_distance_variables(
    vars_collection: VariableCollection,
    mod: gp.Model,
    fg: FactorGraphData,
    relaxation_type: str,
):
    """
    Adds variables to the model that represent the distances between the robot's
    landmarks and the landmarks.

    Args:


    Returns:
        Dict[Tuple[str, str], np.ndarray]: The dict of variables representing
        the distances between the robot's landmarks and the landmarks.
    """
    assert relaxation_type in acceptable_relaxations

    distances: Dict[Tuple[str, str], np.ndarray] = {}
    num_range_measures = len(fg.range_measurements)
    if num_range_measures == 0:
        return distances

    # if we are using the SOCP relaxation then we get all keys and use a single
    # call to instantiate all variables. Otherwise, as the distance variables
    # are matrices, we will have to instantiate them one by one.
    dist_keys = [(meas.pose_key, meas.landmark_key) for meas in fg.range_measurements]
    if relaxation_type == "SOCP":
        dist_vars = mod.addVars(
            dist_keys,
            lb=0,
        )
        vars_collection.set_distance_variables(dist_vars)
    elif relaxation_type == "QCQP":
        for dist_key in dist_keys:
            name = f"dist_{dist_key[0]}_{dist_key[1]}"
            vars_collection.add_distance_variable(
                mod.addMVar(
                    shape=(fg.dimension,),
                    lb=-gp.GRB.INFINITY,
                    ub=gp.GRB.INFINITY,
                    name=name,
                ),
                dist_key,
            )
    else:
        raise ValueError(f"Relaxation type {relaxation_type} not supported")

    # logger.info("Done adding distance variables")
    return distances


##### set up constraints #####


def pin_pose(
    pose_var: gp.MVar,
    mod: gp.Model,
):
    """
    Pins the pose variable to the identity matrix.

    Args:
        pose_var (gp.MVar): The pose variable to pin.
        mod (gp.Model): The model to add the constraint to.
    """
    dim = pose_var.shape[0]
    for i in range(dim):
        for j in range(dim + 1):
            if i == j:
                mod.addConstr(pose_var[i, j] == 1)
            else:
                mod.addConstr(pose_var[i, j] == 0)


def add_distance_constraints(
    variables: VariableCollection,
    mod: gp.Model,
    relaxation_type: str,
):
    if relaxation_type == "QCQP":
        # constrain distance variables to be within unit ball
        for dist_var in variables.distance_vars.values():
            mod.addConstr(vec_norm_sq(dist_var) <= 1)
    if relaxation_type == "SOCP":
        # create second-order cone constraints
        # ||t_i - t_j||^2 <= d_ij^2
        for dist_key, dist_var in variables.distance_vars.items():
            trans_i = variables.get_translation_var(dist_key[0])
            trans_j = variables.get_translation_var(dist_key[1])
            diff = trans_i - trans_j
            mod.addConstr(vec_norm_sq(diff) <= dist_var**2)


##### set up objective #####


def get_full_cost_objective(
    variables: VariableCollection, data: FactorGraphData, relaxation_type: str
) -> gp.MQuadExpr:
    """Get the full cost objective function for the factor graph.

    Args:
        variables (VariableCollection): the variables in the model
        data (FactorGraphData): the factor graph data
        relaxation_type (str): the relaxation type to use

    Returns:
        gp.MQuadExpr: the full cost objective function
    """
    _check_valid_relaxation(relaxation_type)
    cost = 0
    cost += get_all_odom_costs(variables, data)
    cost += get_all_loop_closure_costs(variables, data)
    cost += get_all_range_costs(variables, data, relaxation_type)
    cost += get_all_landmark_prior_costs(variables, data)
    return cost


def get_all_odom_costs(
    variables: VariableCollection,
    data: FactorGraphData,
):
    """Add the cost associated with the odometry measurements as:

        translation component of cost
        k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2

        rotation component of cost
        tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob^2

    Args:
        variables (VariableCollection): the variables in the model
        data (FactorGraphData): the factor graph data

    """
    cost = 0
    for odom_chain in data.odom_measurements:
        for odom_measure in odom_chain:
            pose_i = variables.get_pose_var(odom_measure.base_pose)
            pose_j = variables.get_pose_var(odom_measure.to_pose)
            cost += get_relative_pose_cost_expression(pose_i, pose_j, odom_measure)

    return cost


def get_all_loop_closure_costs(
    variables: VariableCollection,
    data: FactorGraphData,
):
    """Add the cost associated with the loop closure measurements as:

        translation component of cost
        k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2

        rotation component of cost
        tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob^2

    Args:
        variables (VariableCollection): the variables in the model
        data (FactorGraphData): the factor graph data

    """
    cost = 0.0
    for loop_measure in data.loop_closure_measurements:
        pose_i = variables.get_pose_var(loop_measure.base_pose)
        pose_j = variables.get_pose_var(loop_measure.to_pose)
        cost += get_relative_pose_cost_expression(pose_i, pose_j, loop_measure)

    return cost


def get_all_landmark_prior_costs(
    var_collection: VariableCollection,
    data: FactorGraphData,
) -> gp.MQuadExpr:
    cost = 0.0
    for landmark_prior in data.landmark_priors:
        # translation component of cost
        # k_ij * ||t_i - t_ij||^2
        t_i = var_collection.get_translation_var(landmark_prior.name)
        translation_term = t_i - landmark_prior.translation_vector
        unweighted_cost = vec_norm_sq(translation_term)
        weight = landmark_prior.translation_precision
        cost += weight * unweighted_cost
    return cost


def get_all_range_costs(
    variables: VariableCollection, data: FactorGraphData, relaxation: str
) -> gp.MQuadExpr:
    """Add the cost associated with the range measurements as:

        k_ij * ||d_ij - d_ij^meas||^2

    Args:
        variables (VariableCollection): the variables in the model
        data (FactorGraphData): the factor graph data
        relaxation (str): the relaxation type

    """
    cost = 0.0
    for range_measure in data.range_measurements:
        dist_key = (range_measure.pose_key, range_measure.landmark_key)
        cost += get_single_range_cost(
            variables.get_translation_var(dist_key[0]),
            variables.get_translation_var(dist_key[1]),
            variables.get_distance_var(dist_key),
            range_measure,
            relaxation,
        )
    return cost


def get_single_range_cost(
    t_i: gp.MVar,
    t_j: gp.MVar,
    d_ij: gp.MVar,
    measure: FGRangeMeasurement,
    relaxation: str,
) -> gp.QuadExpr:
    _check_valid_relaxation(relaxation)

    # SOCP cost = k_ij * ||d_ij - d_ij^meas||^2
    # QCQP cost = k_ij * ||t_i - t_j - (d_ij^meas) * d_ij||^2
    if relaxation == "SOCP":
        unweighted_cost = measure.dist**2 - 2 * measure.dist * d_ij + d_ij**2
    elif relaxation == "QCQP":
        intermed = t_i - t_j - d_ij * measure.dist
        # print(f"t_i: {t_i}")
        # print(f"t_j: {t_j}")
        # print(f"d_ij: {d_ij}")
        # print(f"ti - tj: {t_i - t_j}")
        # print(f"measure.dist: {measure.dist}")
        # print(f"intermed: {intermed}")
        unweighted_cost = vec_norm_sq(intermed)
    else:
        raise ValueError(f"Relaxation {relaxation} is not supported.")

    cost = measure.precision * unweighted_cost
    return cost


def get_relative_pose_cost_expression(
    pose_i: gp.MVar, pose_j: gp.MVar, measure: POSE_MEASUREMENT_TYPES
) -> gp.QuadExpr:
    t_i = pose_i[:, -1]
    t_j = pose_j[:, -1]
    R_i = pose_i[:, :-1]
    R_j = pose_j[:, :-1]

    # translation component of cost
    # k_ij * ||t_i - t_j - R_i @ t_{ij}^{meas}||^2
    k_ij = measure.translation_precision
    trans_measure = measure.translation_vector
    term = t_j - t_i - (R_i @ trans_measure)
    trans_obj = k_ij * vec_norm_sq(term)

    # rotation component of cost
    # tau_ij * || R_j - (R_i @ R_{ij}^{meas}) ||_\frob
    tau_ij = measure.rotation_precision
    rot_measure = measure.rotation_matrix
    diff_rot_matrix = R_j - (R_i @ rot_measure)
    rot_obj = tau_ij * mat_frob_norm_sq(diff_rot_matrix)

    return rot_obj + trans_obj


if __name__ == "__main__":
    from os.path import expanduser, join

    grid_len = 70
    num_timesteps = 1000
    num_robots = 4

    def _sample_slam_problem() -> FactorGraphData:
        from manhattan.simulator.simulator import ManhattanSimulator, SimulationParams

        show_animation = False
        num_beacons = 0
        range_prob = 0.25
        dist_stddev = 0.1
        pos_stddev = 0.1
        theta_stddev = 0.1
        seed_cnt = 0

        sim_args = SimulationParams(
            num_robots=num_robots,
            num_beacons=num_beacons,
            grid_shape=(grid_len, grid_len),
            y_steps_to_intersection=2,
            x_steps_to_intersection=5,
            cell_scale=1.0,
            range_sensing_prob=range_prob,
            range_sensing_radius=100.0,
            false_range_data_association_prob=0.0,
            outlier_prob=0.0,
            max_num_loop_closures=100,
            loop_closure_prob=0.05,
            loop_closure_radius=20.0,
            false_loop_closure_prob=0.0,
            range_stddev=dist_stddev,
            odom_x_stddev=pos_stddev,
            odom_y_stddev=pos_stddev,
            odom_theta_stddev=theta_stddev,
            loop_x_stddev=pos_stddev,
            loop_y_stddev=pos_stddev,
            loop_theta_stddev=theta_stddev,
            debug_mode=False,
            seed_num=(seed_cnt + 1) * 9999,
            groundtruth_measurements=True,
            # no_loop_pose_idx=[0, 1, 2],
            # exclude_last_n_poses_for_loop_closure=2
        )
        sim = ManhattanSimulator(sim_args)

        if show_animation:
            sim.plot_grid()
            sim.plot_beacons()

        for _ in range(num_timesteps):
            sim.random_step()

            if show_animation:
                sim.plot_robot_states()
                sim.show_plot(animation=True)

        if show_animation:
            sim.close_plot()

        sim._factor_graph.print_summary()
        return sim._factor_graph

    fg = _sample_slam_problem()
    pose_chain_names = fg.get_pose_chain_names()

    curr_iter = 0
    finished_solving = False
    save_dir = expanduser("~/data/test_slam_problem")
    import os

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    from time import perf_counter

    start_time = perf_counter()

    relaxation_type = "QCQP"
    score_results = solve_problem(fg, relaxation_type)

    end_time = perf_counter()
    print(f"Total time ({relaxation_type}): {end_time - start_time} seconds")
