import os
from os.path import join
import sys
import logging, coloredlogs
import subprocess

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
from score.solve_score import solve_score
from score.utils.solver_utils import ScoreSolverParams


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

    results_filepath = goats_file_path.replace(".pkl", "_results.tum")

    # * We have a fancy double-solve approach that fixes the rotations solved
    # for the first time and resolves the problem. Right now it's not perfectly
    # fleshed out but it works really well in practice and has some theoretical
    # similarities to the chordal initialization approach. Flip this flag to
    # test it out
    double_solve = False
    if double_solve:
        intermediate_results_filepath = goats_file_path.replace(
            ".pkl", "_intermediate.pkl"
        )
        solve_score(goats_pyfg, solver_params, intermediate_results_filepath)

        second_solve_params = ScoreSolverParams(
            solver="gurobi",
            verbose=True,
            save_results=True,
            init_technique="double_solve_custom",
            custom_init_file=intermediate_results_filepath,
        )
        solve_score(goats_pyfg, second_solve_params, results_filepath)
    else:
        # Solve the problem and save the results to a TUM file for visualization
        solve_score(goats_pyfg, solver_params, results_filepath)

    # have to do this because of the way we have our save functionality set up
    # for multi-robot scenarios (we enumerate robot trajectories with letters)
    results_filepath_saved = results_filepath.replace(".tum", "_A.tum")

    # Visualize the results using evo
    gt_file = join(data_dir, "gt_traj_A.tum")
    subprocess.run(
        [
            "evo_traj",
            "tum",
            results_filepath_saved,
            "--ref",
            gt_file,
            "-va",
            "--plot",
            "--plot_mode",
            "xy",
        ]
    )
