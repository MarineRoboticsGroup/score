from typing import Optional
import attr
import re
import time

from pydrake.solvers.mathematicalprogram import MathematicalProgram  # type: ignore
from py_factor_graph.factor_graph import FactorGraphData

import ro_slam.utils.drake_utils as du
from ro_slam.utils.plot_utils import plot_error
from ro_slam.utils.solver_utils import (
    QcqpSolverParams,
    SolverResults,
    save_results_to_file,
    load_custom_init_file,
)


def solve_mle_qcqp(
    data: FactorGraphData,
    solver_params: QcqpSolverParams,
    results_filepath: str,
):
    """
    Takes the data describing the problem and returns the MLE solution to the
    poses and landmark positions

    args:
        data (FactorGraphData): the data describing the problem
        solver (str): the solver to use [ipopt, snopt, default]
        verbose (bool): whether to show verbose solver output
        save_results (bool): whether to save the results to a file
        results_filepath (str): the path to save the results to
        use_socp_relax (bool): whether to use socp relaxation on distance
            variables
        use_orthogonal_constraint (bool): whether to use orthogonal
            constraint on rotation variables
    """
    solver_options = ["mosek", "gurobi", "ipopt", "snopt", "default"]
    assert (
        solver_params.solver in solver_options
    ), f"Invalid solver, must be from: {solver_options}"

    init_options = ["gt", "compose", "random", "none", "custom"]
    assert (
        solver_params.init_technique in init_options
    ), f"Invalid init_technique, must be from: {init_options}"

    if solver_params.solver in ["mosek", "gurobi"]:
        assert (
            solver_params.use_socp_relax and not solver_params.use_orthogonal_constraint
        ), "Mosek and Gurobi solver only used to solve convex problems"

    model = MathematicalProgram()

    # form objective function
    translations, rotations = du.add_pose_variables(
        model, data, solver_params.use_orthogonal_constraint
    )
    print("Added pose variables")
    assert (translations.keys()) == (rotations.keys())

    landmarks = du.add_landmark_variables(model, data)
    print("Added landmark variables")
    distances = du.add_distance_variables(
        model, data, translations, landmarks, solver_params.use_socp_relax
    )
    print("Added distance variables")

    du.add_distances_cost(model, distances, data)
    du.add_odom_cost(model, translations, rotations, data)
    du.add_loop_closure_cost(model, translations, rotations, data)

    # pin first pose based on data
    du.pin_first_pose(model, translations["A0"], rotations["A0"], data, 0)
    # du.pin_first_pose(model, translations["B0"], rotations["B0"], data, 1)
    # du.pin_first_pose(model, translations["C0"], rotations["C0"], data, 2)

    du.pin_first_landmark(model, landmarks["L0"], data)

    if solver_params.init_technique == "gt":
        du.set_rotation_init_gt(model, rotations, data)
        du.set_translation_init_gt(model, translations, data)
        du.set_distance_init_gt(model, distances, data)
        du.set_landmark_init_gt(model, landmarks, data)
    elif solver_params.init_technique == "compose":
        du.set_rotation_init_compose(model, rotations, data)
        du.set_translation_init_compose(model, translations, data)
        du.set_distance_init_measured(model, distances, data)
        du.set_landmark_init_random(model, landmarks)
    elif solver_params.init_technique == "random":
        du.set_rotation_init_random(model, rotations)
        du.set_translation_init_random(model, translations)
        du.set_distance_init_random(model, distances)
        du.set_landmark_init_random(model, landmarks)
    elif solver_params.init_technique == "custom":
        assert (
            solver_params.custom_init_file is not None
        ), "Must provide custom_init_filepath if using custom init"
        custom_vals = load_custom_init_file(solver_params.custom_init_file)
        init_rotations = custom_vals.rotations
        init_translations = custom_vals.translations
        init_landmarks = custom_vals.landmarks
        du.set_rotation_init_custom(model, rotations, init_rotations)
        du.set_translation_init_custom(model, translations, init_translations)
        du.set_landmark_init_custom(model, landmarks, init_landmarks)
        du.set_distance_init_valid(model, distances, init_translations, init_landmarks)

    # perform optimization
    print("Starting solver...")

    t_start = time.time()
    try:
        solver = du.get_drake_solver(solver_params.solver)
        if solver_params.verbose:
            du.set_drake_solver_verbose(model, solver)

        if solver_params.solver == "gurobi":
            # model.SetSolverOption(solver.solver_id(), "BarQCPConvTol", 1e-8)
            # model.SetSolverOption(solver.solver_id(), "BarConvTol", 1e-8)
            # model.SetSolverOption(solver.solver_id(), "BarHomogeneous", 1)
            pass

        result = solver.Solve(model)
    except Exception as e:
        print("Error: ", e)
        return
    t_end = time.time()
    tot_time = t_end - t_start
    print(f"Solved in {tot_time} seconds")
    print(f"Solver success: {result.is_success()}")

    solution_vals = du.get_solved_values(
        result,
        tot_time,
        translations,
        rotations,
        landmarks,
        distances,
        data.get_pose_chain_names(),
    )

    if solver_params.save_results:
        save_results_to_file(
            solution_vals,
            result.is_success(),
            result.get_optimal_cost(),
            results_filepath,
        )

    grid_size_search = re.search(r"\d+_grid", results_filepath)
    if grid_size_search is not None:
        grid_size = int(grid_size_search.group(0).split("_")[0])
    else:
        grid_size = 1

    # if solver_params.init_technique == "custom":
    #     plot_error(data, solution_vals, grid_size, custom_vals)
    # else:
    #     # do not solve local so only print the relaxed solution
    #     plot_error(data, solution_vals, grid_size)
