from typing import Union, Dict, Optional, Tuple, List
import pickle
from os.path import isfile, dirname, isdir
from os import makedirs
import numpy as np
import attr


from ro_slam.utils.matrix_utils import (
    get_theta_from_rotation_matrix,
    get_rotation_matrix_from_transformation_matrix,
    get_theta_from_transformation_matrix,
    get_translation_from_transformation_matrix,
    _check_rotation_matrix,
    _check_transformation_matrix,
)


@attr.s(frozen=True)
class QcqpSolverParams:
    solver: str = attr.ib()
    verbose: bool = attr.ib()
    save_results: bool = attr.ib()
    use_socp_relax: bool = attr.ib()
    use_orthogonal_constraint: bool = attr.ib()
    init_technique: str = attr.ib()
    custom_init_file: Optional[str] = attr.ib(default=None)


@attr.s(frozen=True)
class GtsamSolverParams:
    verbose: bool = attr.ib()
    save_results: bool = attr.ib()
    init_technique: str = attr.ib()
    custom_init_file: Optional[str] = attr.ib(default=None)
    init_translation_perturbation: Optional[float] = attr.ib(default=None)
    init_rotation_perturbation: Optional[float] = attr.ib(default=None)


@attr.s(frozen=True)
class VariableValues:
    poses: Dict[str, np.ndarray] = attr.ib()
    # translations: Dict[str, np.ndarray] = attr.ib()
    # rotations: Dict[str, np.ndarray] = attr.ib()
    landmarks: Dict[str, np.ndarray] = attr.ib()
    distances: Optional[Dict[Tuple[str, str], np.ndarray]] = attr.ib()

    @poses.validator
    def _check_poses(self, attribute, value: Dict[str, np.ndarray]):
        for pose in value.values():
            _check_transformation_matrix(pose)

    @property
    def rotations(self):
        return {
            key: get_theta_from_transformation_matrix(value)
            for key, value in self.poses.items()
        }

    @property
    def rotations_matrix(self):
        return {
            key: get_rotation_matrix_from_transformation_matrix(value)
            for key, value in self.poses.items()
        }

    @property
    def translations(self):
        return {
            key: get_translation_from_transformation_matrix(value)
            for key, value in self.poses.items()
        }


@attr.s(frozen=True)
class SolverResults:
    variables: VariableValues = attr.ib()
    total_time: float = attr.ib()
    solved: bool = attr.ib()
    pose_chain_names: Optional[list] = attr.ib(default=None)  # Default [[str]]

    @property
    def poses(self):
        return self.variables.poses

    @property
    def translations(self):
        return self.variables.translations

    @property
    def rotations(self):
        return self.variables.rotations

    @property
    def landmarks(self):
        return self.variables.landmarks

    @property
    def distances(self):
        return self.variables.distances


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
    print(status)


def save_results_to_file(
    solved_results: SolverResults,
    solved_cost: float,
    solve_success: bool,
    filepath: str,
):
    """
    Saves the results to a file

    Args:
        result (Drake Results): the result of the solution
        solved_results (Dict[str, Dict[str, np.ndarray]]): the solved values of the variables
        filepath (str): the path to save the results to
    """
    data_dir = dirname(filepath)
    if not isdir(data_dir):
        makedirs(data_dir)

    if filepath.endswith(".pickle"):
        pickle_file = open(filepath, "wb")
        pickle.dump(solved_results, pickle_file)
        solve_info = {
            "success": solve_success,
            "optimal_cost": solved_cost,
        }
        pickle.dump(solve_info, pickle_file)
        pickle_file.close()

    elif filepath.endswith(".txt"):
        with open(filepath, "w") as f:
            translations = solved_results.translations
            rotations = solved_results.rotations
            for pose_key in translations.keys():
                trans_solve = translations[pose_key]
                theta_solve = rotations[pose_key]

                trans_string = np.array2string(
                    trans_solve, precision=1, floatmode="fixed"
                )
                status = (
                    f"State {pose_key}"
                    + f" | Translation: {trans_string}"
                    + f" | Rotation: {theta_solve:.2f}\n"
                )
                f.write(status)

            landmarks = solved_results.landmarks
            for landmark_key in landmarks.keys():
                landmark_solve = landmarks[landmark_key]

                landmark_string = np.array2string(
                    landmark_solve, precision=1, floatmode="fixed"
                )
                status = (
                    f"State {landmark_key}" + f" | Translation: {landmark_string}\n"
                )
                f.write(status)

            f.write(f"Is optimization successful? {solve_success}\n")
            f.write(f"optimal cost: {solved_cost}")

    # Outputs each posechain as a separate file with timestamp in TUM format
    elif filepath.endswith(".tum"):
        save_to_tum(solved_results, filepath)
    else:
        raise ValueError(
            f"The file extension {filepath.split('.')[-1]} is not supported. "
        )

    print(f"Results saved to: {filepath}\n")


def save_to_tum(
    solved_results: SolverResults, filepath: str, strip_extension: bool = False
):
    """Saves a given set of solver results to a number of TUM files, with one
    for each pose chain in the results.

    Args:
        solved_results (SolverResults): [description]
        filepath (str): the path to save the results to. The final files will
        have the pose chain letter appended to the end to indicate which pose chain.
        strip_extension (bool, optional): Whether to strip the file extension
        and replace with ".tum". This should be set to true if the file
        extension is not already ".tum". Defaults to False.
    """
    assert (
        solved_results.pose_chain_names is not None
    ), "Pose_chain_names must be provided for multi robot trajectories"
    # TODO: Add support for exporting without pose_chain_names
    for pose_chain in solved_results.pose_chain_names:
        if len(pose_chain) == 0:
            continue
        pose_chain_letter = pose_chain[0][0]  # Get first letter of first pose in chain
        assert (
            pose_chain_letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ), "Pose chain letter must be uppercase letter"

        # Removes extension from filepath to add tum extension
        if strip_extension:
            filepath = filepath.split(".")[0] + ".tum"

        assert filepath.endswith(".tum"), "File extension must be .tum"
        modified_path = filepath.replace(".tum", f"_{pose_chain_letter}.tum")

        # if file already exists we won't write over it
        if isfile(modified_path):
            print(f"{modified_path} already exists, skipping")
            continue

        with open(modified_path, "w") as f:
            translations = solved_results.translations
            rotations = solved_results.rotations
            for i, pose_key in enumerate(pose_chain):
                trans_solve = translations[pose_key]
                theta_solve = rotations[pose_key]
                # TODO: Add actual timestamps
                f.write(
                    f"{i} {trans_solve[0]} {trans_solve[1]} 0 0 0 {np.sin(theta_solve/2)} {np.cos(theta_solve/2)}\n"
                )


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


def load_custom_init_file(file_path: str) -> VariableValues:
    """Loads the custom init file

    Args:
        file_path (str): [description]
    """

    assert isfile(file_path), f"File {file_path} does not exist"
    assert file_path.endswith(".pickle"), f"File {file_path} must end with '.pickle'"

    print(f"Loading custom init file: {file_path}")
    with open(file_path, "rb") as f:
        init_dict = pickle.load(f)
        if isinstance(init_dict, SolverResults):
            return init_dict.variables
        elif isinstance(init_dict, VariableValues):
            return init_dict
        else:
            raise ValueError(f"Unknown type: {type(init_dict)}")
