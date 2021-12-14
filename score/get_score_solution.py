from os.path import join

from py_factor_graph.parse_factor_graph import (
    parse_efg_file,
    parse_pickle_file,
)
from ro_slam.solve_mle_qcqp import solve_mle_qcqp, QcqpSolverParams


if __name__ == "__main__":
    solver_params = QcqpSolverParams(
        solver="gurobi",
        verbose=True,
        save_results=True,
        use_socp_relax=True,
        use_orthogonal_constraint=False,
        init_technique="random",
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

    if fg_filepath.endswith(".pickle"):
        fg = parse_pickle_file(fg_filepath)
    elif fg_filepath.endswith(".fg"):
        fg = parse_efg_file(fg_filepath)
    else:
        raise ValueError(f"Unknown file type: {fg_filepath}")
    print(f"Loaded data: {fg_filepath}")
    fg.print_summary()

    # check that the measurements are all good
    # assert fg.only_good_measurements()

    results_filepath = join(args.results_dir, args.results_filename)
    solve_mle_qcqp(fg, solver_params, results_filepath)
