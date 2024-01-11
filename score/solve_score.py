from os.path import join
import logging, coloredlogs
from typing import List

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="WARNING",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import (
    SolverResults,
)
from py_factor_graph.utils.matrix_utils import get_matrix_determinant


import score.utils.gurobi_utils as gu
from gurobipy import GRB, GurobiError


def _check_factor_graph(data: FactorGraphData):
    unconnected_variables = data.unconnected_variable_names
    assert (
        len(unconnected_variables) == 0
    ), f"Found {unconnected_variables} unconnected variables. "


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


def solve_score(
    data: FactorGraphData,
    relaxation_type: str = gu.QCQP_RELAXATION,
) -> SolverResults:
    """
    Takes the data describing the problem and returns the MLE solution to the
    poses and landmark positions

    args:
        data (FactorGraphData): the data describing the problem
        results_filepath (str): where to save the results

    returns:
        SolverResults: the results of the solver
    """

    _check_factor_graph(data)

    variables = gu.VariableCollection(data.dimension)
    model = gu.get_model()
    gu.initialize_model(variables, model, data, relaxation_type)
    try:
        model.optimize()
    except GurobiError as e:
        # set the nonconvex parameter to 2 and try again
        logger.warning(
            f"GurobiError: {e}. "
            "If the problem is just slightly non-convex, this may be because of numerical issues. "
            "Resolving with nonconvex parameter set to 2."
        )
        model.setParam(GRB.Param.NonConvex, 2)
        model.optimize()
    return gu.extract_solver_results(model, variables, data)


def solve_problem_with_intermediate_iterates(
    data: FactorGraphData, relaxation_type: str
) -> List[SolverResults]:
    logger.warning(
        """Solving the problem with intermediate iterates - this is for
        debugging or visualization only as it is much slower than a single
        solve. Use solve_score() for solving the problem"""
    )
    iterates = []
    model_vars = gu.VariableCollection(dim=data.dimension)
    model = gu.get_model()
    gu.initialize_model(model_vars, model, data, relaxation_type)
    curr_iter = 0
    finished_solving = False
    while not finished_solving:
        model.Params.BarIterLimit = curr_iter
        model.optimize()

        # check if solver done
        if model.status != GRB.Status.ITERATION_LIMIT:
            finished_solving = True

        curr_solver_results = gu.extract_solver_results(model, model_vars, data)
        iterates.append(curr_solver_results)

        curr_iter += 1

    return iterates
