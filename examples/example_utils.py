import os
from os.path import join
from typing import List, Tuple
import re

from ro_slam.utils.solver_utils import QcqpSolverParams, GtsamSolverParams


def get_folders_in_dir(path) -> List[str]:
    return [join(path, f) for f in os.listdir(path) if os.path.isdir(join(path, f))]


def get_files_in_dir(path) -> List[str]:
    return [join(path, f) for f in os.listdir(path) if os.path.isfile(join(path, f))]


def recursively_find_pickle_files(dir) -> List[Tuple[str, str]]:
    """Recursively finds all .pickle files in the directory and its subdirectories

    Args:
        dir (str): the directory to search in

    Returns:
        List[Tuple[str, str]]: a list of tuples of the form (root, file_name)

    """

    # def num_timesteps_from_path(path: str) -> int:
    #     trailing_phrase = "_timesteps"
    #     info = re.search(r"\d+" + trailing_phrase, path).group(0)  # type: ignore
    #     num_timesteps = int(info[: -len(trailing_phrase)])
    #     return num_timesteps

    pickle_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".pickle"):
                pickle_files.append((root, file))

    # pickle_files.sort(key=lambda x: num_timesteps_from_path(x[0]))
    return pickle_files


def get_factor_graph_file_in_dir(path) -> str:

    # prefer pickle files but also check for .fg files
    pickle_files = [x for x in get_files_in_dir(path) if x.endswith(".pickle")]
    if len(pickle_files) >= 1:
        assert (
            len(pickle_files) == 1
        ), "There should be only one pickle file in the directory"
        return pickle_files[0]

    efg_files = [x for x in get_files_in_dir(path) if x.endswith(".fg")]
    if len(efg_files) >= 1:
        assert (
            len(efg_files) == 1
        ), "There should be only one factor graph file in the directory"
        return efg_files[0]

    raise ValueError(f"No factor graph file found in the directory: {path}")


def get_qcqp_results_filename(
    solver_params: QcqpSolverParams,
    filetype: str = "pickle",
) -> str:
    """Returns the name of the results file

    Args:
        solver_params (QcqpSolverParams): the solver parameters

    Returns:
        str: the file name giving details of the solver params
    """
    file_name = f"{solver_params.solver}_"

    file_name += f"init{solver_params.init_technique}_"

    # add in indicator for SOCP relaxation
    if solver_params.use_socp_relax:
        file_name += "socp"
    else:
        file_name += "nosocp"
    file_name += "_"

    # add in indicator for orthogonal constraints
    if solver_params.use_orthogonal_constraint:
        file_name += "orth"
    else:
        file_name += "noorth"
    file_name += "_results."

    # add in results.txt and return
    file_name += filetype
    return file_name


def get_gtsam_results_filename(
    solver_params: GtsamSolverParams,
    filetype: str = "pickle",
) -> str:
    """Returns the name of the results file

    Args:
        solver_params (GtsamSolverParams): the solver parameters

    Returns:
        str: the file name giving details of the solver params
    """
    file_name = "{gtsam}_"

    file_name += f"init{solver_params.init_technique}_"

    file_name += "_results."

    # add in results.txt and return
    file_name += filetype
    return file_name
