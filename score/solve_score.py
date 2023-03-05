from os.path import join
import time

import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)

from pydrake.solvers.mathematicalprogram import MathematicalProgram  # type: ignore
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file
from py_factor_graph.parsing.parse_efg_file import parse_efg_file
from py_factor_graph.utils.solver_utils import (
    SolverResults,
    save_results_to_file,
    load_custom_init_file,
)
from py_factor_graph.utils.matrix_utils import get_matrix_determinant


import score.utils.drake_utils as du
from score.utils.solver_utils import (
    QcqpSolverParams,
)


def _check_solver_params(solver_params: QcqpSolverParams):
    if solver_params.solver in ["mosek", "gurobi"]:
        assert (
            solver_params.use_socp_relax and not solver_params.use_orthogonal_constraint
        ), "Mosek and Gurobi solver only used to solve convex problems"


def _check_factor_graph(data: FactorGraphData):
    unconnected_variables = data.unconnected_variable_names
    assert (
        len(unconnected_variables) == 0
    ), f"Found {unconnected_variables} unconnected variables. "


def _initialize_variables(
    model,
    data: FactorGraphData,
    solver_params: QcqpSolverParams,
    rotations,
    translations,
    distances,
    landmarks,
):

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
        du.set_rotation_init_random(model, rotations, data)
        du.set_translation_init_random(model, translations, data)
        du.set_distance_init_random(model, distances)
        du.set_landmark_init_random(model, landmarks)
    elif solver_params.init_technique == "custom":
        assert (
            solver_params.custom_init_file is not None
        ), "Must provide custom_init_filepath if using custom init"
        custom_vals = load_custom_init_file(solver_params.custom_init_file)
        init_rotations = custom_vals.rotations_matrix
        init_translations = custom_vals.translations
        init_landmarks = custom_vals.landmarks
        du.set_rotation_init_custom(model, rotations, init_rotations)
        du.set_translation_init_custom(model, translations, init_translations)
        du.set_landmark_init_custom(model, landmarks, init_landmarks)
        du.set_distance_init_valid(model, distances, init_translations, init_landmarks)
    elif solver_params.init_technique == "double_solve_custom":
        assert (
            solver_params.custom_init_file is not None
        ), "Must provide custom_init_filepath if using custom init"
        custom_vals = load_custom_init_file(solver_params.custom_init_file)
        init_rotations = custom_vals.rotations_matrix
        init_translations = custom_vals.translations
        init_landmarks = custom_vals.landmarks
        du.set_rotation_init_custom(model, rotations, init_rotations)
        du.constrain_rotations_to_custom(model, rotations, init_rotations)
        du.set_translation_init_custom(model, translations, init_translations)
        du.set_landmark_init_custom(model, landmarks, init_landmarks)
        du.set_distance_init_valid(model, distances, init_translations, init_landmarks)


def _solve_problem(model, solver_params: QcqpSolverParams):
    print("Starting solver...")

    t_start = time.time()
    try:
        solver = du.get_drake_solver(solver_params.solver)
        if solver_params.verbose:
            du.set_drake_solver_verbose(model, solver)

        if solver_params.solver == "gurobi":
            # model.SetSolverOption(solver.solver_id(), "BarQCPConvTol", 1e-12)
            # model.SetSolverOption(solver.solver_id(), "BarConvTol", 1e-12)
            model.SetSolverOption(solver.solver_id(), "BarHomogeneous", 1)

            # set max number of iterations
            if solver_params.iterations is not None:
                model.SetSolverOption(
                    solver.solver_id(), "BarIterLimit", solver_params.iterations
                )

            # Set to be numerically conservative:
            # https://www.gurobi.com/documentation/9.5/refman/numericfocus.html
            # model.SetSolverOption(solver.solver_id(), "NumericFocus", 3)
            # pass

        result = solver.Solve(model)
    except Exception as e:
        print("Error: ", e)
        raise e
    t_end = time.time()
    tot_time = t_end - t_start
    print(f"Solved in {tot_time} seconds")
    print(f"Solver success: {result.is_success()}")

    return result, tot_time


def _check_solution_quality(result, rotations):
    # get list of the determinants of the rotation matrices
    det_list = [
        get_matrix_determinant(result.GetSolution(rotations[key]))
        for key in rotations.keys()
    ]

    import matplotlib.pyplot as plt

    logger.warning(
        "Plotting the rotation matrix determinants - be sure to close the plot to continue"
    )
    x_idxs = [i for i in range(len(det_list))]
    plt.plot(x_idxs, det_list)
    plt.ylim([-0.1, 1.1])
    plt.title("Determinants of Unrounded Rotation Matrices")
    plt.show(block=True)  # type: ignore


def solve_mle_qcqp(
    data: FactorGraphData,
    solver_params: QcqpSolverParams,
    results_filepath: str,
) -> SolverResults:
    """
    Takes the data describing the problem and returns the MLE solution to the
    poses and landmark positions

    args:
        data (FactorGraphData): the data describing the problem
        solver_params (QcqpSolverParams): the parameters for the solver
        results_filepath (str): where to save the results

    returns:
        SolverResults: the results of the solver
    """

    _check_solver_params(solver_params)
    _check_factor_graph(data)
    logger.debug(f"Running SCORE solver with params: {solver_params}")

    model = MathematicalProgram()

    # Add variables
    translations, rotations = du.add_pose_variables(
        model, data, solver_params.use_orthogonal_constraint
    )
    landmarks = du.add_landmark_variables(model, data)
    distances = du.add_distance_variables(
        model, data, translations, landmarks, solver_params.use_socp_relax
    )

    # initialize the variables based on the solver params
    _initialize_variables(
        model, data, solver_params, rotations, translations, distances, landmarks
    )

    # Add costs
    du.add_distances_cost(model, distances, data)
    du.add_odom_cost(model, translations, rotations, data)
    du.add_loop_closure_cost(model, translations, rotations, data)
    du.add_pose_prior_cost(model, translations, rotations, data)
    du.add_landmark_prior_cost(model, landmarks, data)

    # pin first pose based on data
    du.pin_first_pose(model, translations["A0"], rotations["A0"], data, 0)

    # perform optimization
    result, tot_time = _solve_problem(model, solver_params)

    #! check the quality of the solution via the matrix determinants
    # _check_solution_quality(result, rotations)

    solution_vals = du.get_solved_values(
        result=result,
        dim=data.dimension,
        time=tot_time,
        translations=translations,
        rotations=rotations,
        landmarks=landmarks,
        distances=distances,
        pose_chain_names=data.get_pose_chain_names(),
    )

    if solver_params.save_results:
        save_results_to_file(
            solution_vals,
            result.is_success(),
            result.get_optimal_cost(),
            results_filepath,
        )

    return solution_vals


if __name__ == "__main__":
    solver_params = QcqpSolverParams(
        solver="gurobi",
        verbose=True,
        save_results=True,
        use_socp_relax=True,
        use_orthogonal_constraint=False,
        init_technique="gt",
        custom_init_file=None,
    )

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "data_dir", type=str, help="Path to the directory the PyFactorGraph is held in"
    )
    arg_parser.add_argument("pyfg_filename", type=str, help="name of the PyFactorGraph")
    arg_parser.add_argument(
        "results_dir", type=str, help="Path to the directory the results are saved to"
    )
    arg_parser.add_argument(
        "results_filename", type=str, help="name of the results file"
    )
    args = arg_parser.parse_args()

    # get the factor graph filepath
    results_filetype = "pickle"
    fg_filepath = join(args.data_dir, args.pyfg_filename)

    if fg_filepath.endswith(".pickle") or fg_filepath.endswith(".pkl"):
        fg = parse_pickle_file(fg_filepath)
    elif fg_filepath.endswith(".fg"):
        fg = parse_efg_file(fg_filepath)
    else:
        raise ValueError(f"Unknown file type: {fg_filepath}")
    logger.info(f"Loaded data: {fg_filepath}")
    fg.print_summary()

    double_solve = False
    if double_solve:
        # round the measurements
        logger.warning("Using the double solve approach to rounding")

        results_filepath = join(args.results_dir, "intermediate.pickle")
        solve_mle_qcqp(fg, solver_params, results_filepath)

        second_solve_params = QcqpSolverParams(
            solver="gurobi",
            verbose=True,
            save_results=True,
            use_socp_relax=True,
            use_orthogonal_constraint=False,
            init_technique="double_solve_custom",
            custom_init_file=results_filepath,
        )
        second_results_filepath = join(args.results_dir, args.results_filename)
        solve_mle_qcqp(fg, second_solve_params, second_results_filepath)
    else:
        logger.warning("Using the single solve approach to rounding")
        results_filepath = join(args.results_dir, args.results_filename)
        solve_mle_qcqp(fg, solver_params, results_filepath)
