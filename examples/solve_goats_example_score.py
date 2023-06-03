import os
from os.path import join
import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)

from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file
from py_factor_graph.utils.plot_utils import visualize_solution
from score.solve_score import solve_score
from score.utils.solver_utils import ScoreSolverParams
from score.utils.gurobi_utils import QCQP_RELAXATION


if __name__ == "__main__":

    # Set up the solver
    solver_params = ScoreSolverParams(
        solver="gurobi",
        verbose=True,
        save_results=True,
        init_technique="none",
        custom_init_file=None,
    )

    # extract the factor graph data
    cur_file_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = join(cur_file_dir, "goats_14_data")
    goats_file_path = join(data_dir, "goats_14_6_2002_15_20.pkl")
    goats_pyfg = parse_pickle_file(goats_file_path)

    score_result = solve_score(
        goats_pyfg, solver_params, QCQP_RELAXATION
    )  # the solution to the convex relaxation - not the refined result!
    visualize_solution(score_result)
