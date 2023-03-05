from typing import Dict, Optional
import numpy as np
import attr
import logging

logger = logging.getLogger(__name__)

from py_factor_graph.utils.matrix_utils import (
    get_theta_from_rotation_matrix,
    _check_rotation_matrix,
)


@attr.s(frozen=True)
class ScoreSolverParams:
    solver: str = attr.ib()
    verbose: bool = attr.ib()
    save_results: bool = attr.ib()
    init_technique: str = attr.ib(default="random")
    custom_init_file: Optional[str] = attr.ib(default=None)
    iterations: Optional[int] = attr.ib(default=None)

    @solver.validator
    def _check_solver(self, attribute, value):
        solver_options = ["mosek", "gurobi", "ipopt", "snopt", "default"]
        assert (
            value in solver_options
        ), f"Invalid solver ({value}) must be from: {solver_options}"

    @init_technique.validator
    def _check_init_technique(self, attribute, value):
        init_options = [
            "gt",
            "compose",
            "random",
            "none",
            "custom",
            "double_solve_custom",
        ]
        assert (
            value in init_options
        ), f"Invalid init_technique, must be from: {init_options}"


def print_state(
    result,
    translations: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    pose_key: str,
):
    """
    Prints the current state of the result

    Args:
        result (MathematicalProgram): the result of the solution
        translations (List[np.ndarray]): the translations
        rotations (List[np.ndarray]): the rotations
        pose_key (str): the key of the pose to print
    """
    trans_solve = result.GetSolution(translations[pose_key]).round(decimals=2)
    rot_solve = result.GetSolution(rotations[pose_key])
    theta_solve = get_theta_from_rotation_matrix(rot_solve)

    trans_string = np.array2string(trans_solve, precision=1, floatmode="fixed")

    status = (
        f"State {pose_key}"
        + f" | Translation: {trans_string}"
        + f" | Rotation: {theta_solve:.2f}"
    )
    logger.info(status)


def check_rotations(result, rotations: Dict[str, np.ndarray]):
    """
    checks that results are valid rotations

    Args:
        result (Drake Result Object): the result of the solution
        rotations (Dict[str, np.ndarray]): the rotation variables
    """
    for rot_key in rotations.keys():
        rot_result = result.GetSolution(rotations[rot_key])
        _check_rotation_matrix(rot_result)
