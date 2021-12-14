import os
from os.path import join, expanduser, abspath, realpath
import sys


file_dir = os.path.dirname(os.path.realpath(__file__))
ro_slam_dir = abspath(realpath(join(file_dir, "..")))
sys.path.insert(0, ro_slam_dir)

from py_factor_graph.parse_factor_graph import (
    parse_efg_file,
    parse_pickle_file,
)
from ro_slam.solve_mle_qcqp import solve_mle_qcqp, QcqpSolverParams
from example_utils import recursively_find_pickle_files, get_qcqp_results_filename


if __name__ == "__main__":
    base_dir = expanduser(join("~", "data", "manhattan"))
    solver_params = QcqpSolverParams(
        solver="gurobi",
        verbose=True,
        save_results=True,
        use_socp_relax=True,
        use_orthogonal_constraint=False,
        init_technique="none",
        custom_init_file=None,
    )
    results_filetype = "pickle"

    # do a recursive search and then test on all of the .pickle files found
    pickle_files = recursively_find_pickle_files(base_dir)
    for pickle_dir, pickle_file in pickle_files:

        if not pickle_file == "factor_graph.pickle":
            continue

        # get the factor graph filepath
        fg_filepath = join(pickle_dir, pickle_file)

        # get the file name to save results to
        results_file_name = get_qcqp_results_filename(solver_params, results_filetype)
        results_filepath = join(pickle_dir, results_file_name)

        # if "100_timesteps" not in fg_filepath:
        #     continue
        # if "3_beacons" not in fg_filepath:
        #     continue
        # if "50_loop"not  in fg_filepath:
        #     continue
        # if "100_loop"not  in fg_filepath:
        #     continue

        if fg_filepath.endswith(".pickle"):
            fg = parse_pickle_file(fg_filepath)
        elif fg_filepath.endswith(".fg"):
            fg = parse_efg_file(fg_filepath)
        else:
            raise ValueError(f"Unknown file type: {fg_filepath}")
        print(f"Loaded data: {fg_filepath}")

        solve_mle_qcqp(fg, solver_params, results_filepath)
    print()
