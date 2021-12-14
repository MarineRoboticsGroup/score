import numpy as np
from typing import List, Tuple, Union, Dict
import tqdm  # type: ignore

from py_factor_graph.factor_graph import FactorGraphData
from ro_slam.utils.matrix_utils import (
    _check_square,
    get_random_vector,
    get_random_rotation_matrix,
    get_rotation_matrix_from_theta,
    round_to_special_orthogonal,
    make_transformation_matrix,
)
from ro_slam.utils.solver_utils import SolverResults, VariableValues

from pydrake.solvers.mathematicalprogram import MathematicalProgram, QuadraticConstraint  # type: ignore
from pydrake.solvers.mathematicalprogram import MathematicalProgramResult as DrakeResult  # type: ignore
from pydrake.solvers.ipopt import IpoptSolver  # type: ignore
from pydrake.solvers.snopt import SnoptSolver  # type: ignore
from pydrake.solvers.gurobi import GurobiSolver  # type: ignore
from pydrake.solvers.mosek import MosekSolver  # type: ignore
from pydrake.solvers.mixed_integer_rotation_constraint import (  # type: ignore
    MixedIntegerRotationConstraintGenerator as MIRCGenerator,
)

# from pydrake.solvers.mixed_integer_optimization_util import IntervalBinning  # type: ignore

##### Add variables #####


def add_pose_variables(
    model: MathematicalProgram, data: FactorGraphData, orthogonal_constraint: bool
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Adds variables to the model that represent the pose of the robot.
    The variables are added to the model in the order of the pose_variables.

    Args:
        model (MathematicalProgram): The model to add the variables to.
        pose_variables (List[PoseVariable]): The list of pose variables to add.
        orthogonal_constraint (bool): Whether to add orthogonal constraints to the rotation matrices (i.e. R.T @ R = I)

    Returns:
        Dict[str, np.ndarray]: The variables representing the translations of the robot keyed by the variable name.
        Dict[str, np.ndarray]: The variables representing the rotations of the the robot keyed by the variable name.
    """
    translations: Dict[str, np.ndarray] = {}
    rotations: Dict[str, np.ndarray] = {}

    print("Adding pose variables")
    for pose_chain in data.pose_variables:
        num_poses = len(pose_chain)
        for pose_idx in tqdm.trange(num_poses):
            pose = pose_chain[pose_idx]

            # add new translation variables d-dimensional vector
            pose_name = pose.name
            trans_name = f"{pose_name}_translation"
            trans_var = add_translation_var(model, trans_name, data.dimension)
            translations[pose_name] = trans_var

            rot_name = f"{pose_name}_rotation"
            rot_var = add_rotation_var(model, rot_name, data.dimension)
            rotations[pose_name] = rot_var

            # TODO test out more efficient constraints?
            # add in rotation constraint (i.e. matrix must be in orthogonal group)
            if orthogonal_constraint:
                set_orthogonal_constraint(model, rot_var)

    return translations, rotations


def add_landmark_variables(
    model: MathematicalProgram, data: FactorGraphData
) -> Dict[str, np.ndarray]:
    """
    Adds variables to the model that represent the landmarks of the robot.

    Args:
        model (MathematicalProgram): The model to add the variables to.
        data (FactorGraphData): The factor graph data.

    Returns:
        Dict[str, np.ndarray]: The variables representing the landmarks of the
        robot keyed by landmark name
    """
    landmarks: Dict[str, np.ndarray] = {}
    num_landmarks = len(data.landmark_variables)
    print("Adding landmarks")
    for landmark_idx in tqdm.trange(num_landmarks):
        landmark = data.landmark_variables[landmark_idx]
        name = f"{landmark.name}_translation"
        landmark_var = add_translation_var(model, name, data.dimension)
        landmarks[landmark.name] = landmark_var
    return landmarks


def add_distance_variables(
    model: MathematicalProgram,
    data: FactorGraphData,
    translations: Dict[str, np.ndarray],
    landmarks: Dict[str, np.ndarray],
    socp_relax: bool,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Adds variables to the model that represent the distances between the robot's
    landmarks and the landmarks.

    Args:
        model (MathematicalProgram): The model to add the variables to.
        data (FactorGraphData): The factor graph data.
        translations (Dict[str, np.ndarray]): The variables representing the translations of the robot
        rotations (Dict[str, np.ndarray]): The variables representing the rotations of the robot
        socp_relax (bool): Whether to relax the distance constraint to SOCP

    Returns:
        Dict[Tuple[str, str], np.ndarray]: The dict of variables representing
        the distances between the robot's landmarks and the landmarks.
    """
    distances = {}
    num_range_measures = len(data.range_measurements)
    print("Adding distance variables")
    for range_measure_idx in tqdm.trange(num_range_measures):
        range_measure = data.range_measurements[range_measure_idx]
        pose_key = range_measure.pose_key
        landmark_key = range_measure.landmark_key

        # create distance variable
        dist_key = (pose_key, landmark_key)
        # name = f"d_p{pose_"
        name = f"d_p{pose_key}_l{landmark_key}"
        distances[dist_key] = add_distance_var(model, name)

        # create distance constraint
        # ||t_i - t_j||^2 <= d_ij^2
        trans_i = translations[pose_key]

        if landmark_key in landmarks:
            trans_j = landmarks[landmark_key]
        elif landmark_key in translations:
            trans_j = translations[landmark_key]
        else:
            raise ValueError(f"Key: {landmark_key} not in landmarks or translations")

        diff = trans_i - trans_j

        # give two options for how to implement the distance constraint
        if socp_relax:
            cone_vars = np.asarray([distances[dist_key][0], *diff.flatten()])
            cone_const = model.AddLorentzConeConstraint(cone_vars)
        else:
            # nonconvex quadratic constraint
            add_drake_distance_equality_constraint(
                model, trans_i, trans_j, distances[dist_key]
            )

    return distances


def add_distance_var(model: MathematicalProgram, name: str) -> np.ndarray:
    """
    Add a variable to the model.

    Args:
        model (MathematicalProgram): The model to add the variable to.
        name (str): The name of the variable.

    Returns:
        np.ndarray: The variable.
    """
    dist_var = model.NewContinuousVariables(1, name=name)
    model.AddBoundingBoxConstraint(0, np.inf, dist_var)
    return dist_var


def add_rotation_var(model: MathematicalProgram, name: str, dim: int) -> np.ndarray:
    """
    Adds a variable to the model that represents the rotation
    matrix of a single pose.

    Args:
        model (MathematicalProgram): The model to add the variable to.
        name (str): The name of the variable.
        dim (int): The dimension of the ambient space

    Returns:
        np.ndarray: The variable representing the rotation of the robot
    """
    rot_var = model.NewContinuousVariables(rows=dim, cols=dim, name=name)
    model.AddBoundingBoxConstraint(-1, 1, rot_var)
    return rot_var


def add_translation_var(model: MathematicalProgram, name: str, dim: int) -> np.ndarray:
    """
    Adds a variable to the model that represents a translation vector.

    Args:
        model (MathematicalProgram): The gurobi model to add the variable to.
        name (str): The name of the variable.
        dim (int): The dimension of the translation vector.

    Returns:
        np.ndarray: The variable representing the translation component of the robot.
    """
    assert dim == 2 or dim == 3
    var = model.NewContinuousVariables(rows=dim, name=name)
    model.AddBoundingBoxConstraint(-np.inf, np.inf, var)
    return var


##### Add costs #####


def add_distances_cost(
    model: MathematicalProgram,
    distances: Dict[Tuple[str, str], np.ndarray],
    data: FactorGraphData,
):
    """Adds in the cost due to the distances as:
    sum_{i,j} k_ij * ||d_ij - d_ij^meas||^2

    Args:
        model (MathematicalProgram): the model to add the cost to
        distances (Dict[Tuple[str, str], np.ndarray]): [description]
        data (FactorGraphData): [description]

    """
    for range_measure in data.range_measurements:
        pose_key = range_measure.pose_key
        landmark_key = range_measure.landmark_key

        # create distance variable
        dist_key = (pose_key, landmark_key)

        # add in distance cost component
        # k_ij * ||d_ij - d_ij^meas||^2
        dist_diff = distances[dist_key] - range_measure.dist

        model.AddQuadraticCost(range_measure.weight * (dist_diff ** 2).sum())


def add_odom_cost(
    model: MathematicalProgram,
    translations: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    data: FactorGraphData,
):
    """Add the cost associated with the odometry measurements as:

        translation component of cost
        k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2

        rotation component of cost
        tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob^2

    Args:
        model (MathematicalProgram): the model to add the cost to
        translations (Dict[str, np.ndarray]): the variables representing translations
        rotations (Dict[str, np.ndarray]): the variables representing rotations
        data (FactorGraphData): the factor graph data

    """
    for odom_chain in data.odom_measurements:
        for odom_measure in odom_chain:

            # the indices of the related poses in the odometry measurement
            i_key = odom_measure.base_pose
            j_key = odom_measure.to_pose

            # get the translation and rotation variables
            t_i = translations[i_key]
            t_j = translations[j_key]
            R_i = rotations[i_key]
            R_j = rotations[j_key]

            # translation component of cost
            # k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2
            trans_weight = odom_measure.translation_weight
            trans_measure = odom_measure.translation_vector
            term = t_j - t_i - (R_i @ trans_measure)
            model.AddQuadraticCost(trans_weight * (term ** 2).sum(), is_convex=True)

            # rotation component of cost
            # tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob
            rot_weight = odom_measure.rotation_weight
            rot_measure = odom_measure.rotation_matrix
            diff_rot_matrix = R_j - (R_i @ rot_measure)
            model.AddQuadraticCost(
                rot_weight * (diff_rot_matrix ** 2).sum(), is_convex=True
            )


def add_loop_closure_cost(
    model: MathematicalProgram,
    translations: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    data: FactorGraphData,
):
    """Add the cost associated with the loop closure measurements as:

        translation component of cost
        k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2

        rotation component of cost
        tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob^2

    Args:
        model (MathematicalProgram): the model to add the cost to
        translations (Dict[str, np.ndarray]): the variables representing translations
        rotations (Dict[str, np.ndarray]): the variables representing rotations
        data (FactorGraphData): the factor graph data

    """
    for loop_measure in data.loop_closure_measurements:

        # the indices of the related poses in the odometry measurement
        i_key = loop_measure.base_pose
        j_key = loop_measure.to_pose

        # get the translation and rotation variables
        t_i = translations[i_key]
        t_j = translations[j_key]
        R_i = rotations[i_key]
        R_j = rotations[j_key]

        # translation component of cost
        # k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2
        trans_weight = loop_measure.translation_weight
        trans_measure = loop_measure.translation_vector
        term = t_j - t_i - (R_i @ trans_measure)
        model.AddQuadraticCost(trans_weight * (term ** 2).sum(), is_convex=True)

        # rotation component of cost
        # tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob
        rot_weight = loop_measure.rotation_weight
        rot_measure = loop_measure.rotation_matrix
        diff_rot_matrix = R_j - (R_i @ rot_measure)
        model.AddQuadraticCost(
            rot_weight * (diff_rot_matrix ** 2).sum(), is_convex=True
        )


##### Initialization strategies #####


def set_distance_init_gt(
    model: MathematicalProgram,
    distances: Dict[Tuple[str, str], np.ndarray],
    data: FactorGraphData,
):
    """Initialize the distance variables to the ground truth distances.

    Args:
        distances (Dict[Tuple[str, str], np.ndarray]): [description]
        data (FactorGraphData): [description]
    """
    print("Setting distance initial points to measured distance")
    for range_measure in data.range_measurements:
        pose_key = range_measure.pose_key
        landmark_key = range_measure.landmark_key
        dist_key = (pose_key, landmark_key)
        model.SetInitialGuess(distances[dist_key][0], range_measure.dist)


def set_distance_init_measured(
    model: MathematicalProgram,
    distances: Dict[Tuple[str, str], np.ndarray],
    data: FactorGraphData,
):
    """Initialize the distance variables to the measured distances.

    Args:
        model (MathematicalProgram): the optimization model
        distances (Dict[Tuple[str, str], np.ndarray]): the distance variables
        data (FactorGraphData): the factor graph data
    """
    print("Setting distance initial points to measured distance")
    for range_measure in data.range_measurements:
        pose_key = range_measure.pose_key
        landmark_key = range_measure.landmark_key
        dist_key = (pose_key, landmark_key)
        model.SetInitialGuess(distances[dist_key][0], range_measure.dist)


def set_distance_init_random(
    model: MathematicalProgram,
    distances: Dict[Tuple[str, str], np.ndarray],
):
    """random initial guess for the distance variables.

    Args:
        model (MathematicalProgram): the optimization model
        distances (Dict[Tuple[str, str], np.ndarray]): the distance variables
    """
    print("Setting distance initial points to random")
    for dist_key in distances:
        model.SetInitialGuess(distances[dist_key], get_random_vector(dim=1))


def init_rotation_variable(
    model: MathematicalProgram, rot: np.ndarray, mat: np.ndarray
):
    """
    Initialize the rotation variables to the given rotation matrix.

    Args:
        rot (np.ndarray): The rotation variables.
        mat (np.ndarray): The rotation matrix.
    """
    assert rot.shape == mat.shape
    model.SetInitialGuess(rot, mat)


def init_translation_variable(
    model: MathematicalProgram, trans: np.ndarray, vec: np.ndarray
):
    """Initialize the translation variables to the given vector

    Args:
        trans (np.ndarray): the variables to initialize
        vec (np.ndarray): the vector
    """
    assert trans.shape == vec.shape, f"trans shape: {trans.shape} vec shape {vec.shape}"
    model.SetInitialGuess(trans, vec)


def set_rotation_init_compose(
    model: MathematicalProgram, rotations: Dict[str, np.ndarray], data: FactorGraphData
) -> None:
    """initializes the rotations by composing the rotations along the odometry chain

    Args:
        rotations (List[np.ndarray]): the rotation variables to initialize
        data (FactorGraphData): the data to use to initialize the rotations
    """
    print("Setting rotation initial points by pose composition")

    # iterate over measurements and init the rotations
    for odom_chain in data.odom_measurements:

        # initialize the first rotation to the identity matrix
        curr_pose = np.eye(data.dimension)
        first_pose_name = odom_chain[0].base_pose
        init_rotation_variable(model, rotations[first_pose_name], curr_pose)

        for odom_measure in odom_chain:

            # update the rotation and initialize the next rotation
            curr_pose = odom_measure.rotation @ curr_pose
            init_rotation_variable(model, rotations[odom_measure.to_pose], curr_pose)


def set_rotation_init_gt(
    model: MathematicalProgram,
    rotations: Dict[str, np.ndarray],
    data: FactorGraphData,
) -> None:
    """Initialize the translation and rotation variables to the ground truth translation
    variables.

    Args:
        model (MathematicalProgram): the model to initialize the variables in
        rotations (Dict[str, np.ndarray]): the rotation variables to initialize
        data (FactorGraphData): the data to use to initialize the variables
    """
    print("Setting rotation initial points to ground truth")
    for pose_chain in data.pose_variables:
        for pose in pose_chain:
            pose_key = pose.name
            rotation_var = rotations[pose_key]
            true_rotation = pose.rotation_matrix
            init_rotation_variable(model, rotation_var, true_rotation)


def set_rotation_init_random(
    model: MathematicalProgram, rotations: Dict[str, np.ndarray]
) -> None:
    """Initializes the rotation variables to random.

    Args:
        model (MathematicalProgram): the model to initialize the variables in
        rotations (Dict[str, np.ndarray]): the rotation variables to initialize
    """
    print("Setting rotation initial points to random")
    for pose_key in rotations:
        rand_rot = get_random_rotation_matrix()
        init_rotation_variable(model, rotations[pose_key], rand_rot)


def set_translation_init_gt(
    model: MathematicalProgram,
    translations: Dict[str, np.ndarray],
    data: FactorGraphData,
) -> None:
    """Initialize the translation and rotation variables to the ground truth translation
    variables.
    """
    print("Setting translation initial points to ground truth")
    for pose_chain in data.pose_variables:
        for pose in pose_chain:
            pose_key = pose.name
            translation_var = translations[pose_key]
            true_translation = np.asarray(pose.true_position)
            init_translation_variable(model, translation_var, true_translation)


def set_translation_init_compose(
    model: MathematicalProgram,
    translations: Dict[str, np.ndarray],
    data: FactorGraphData,
) -> None:
    """Initialize the translation variables by composing the translation
    variables along the odometry chain.

    Args:
        model (MathematicalProgram): the model to initialize the variables in
        translations (Dict[str, np.ndarray]): the translation variables to initialize
        data (FactorGraphData): the data to use to initialize the translation
    """
    print("Setting translation initial points by pose composition")

    # initialize the first transformation to the identity matrix
    curr_pose = np.eye(data.dimension + 1)
    curr_trans = curr_pose[:-1, -1]

    pose0_key = data.pose_variables[0].name
    init_translation_variable(model, translations[pose0_key], curr_trans)

    to_pose_key = ""
    # compose the transformations along the odometry chain
    for odom_measure in data.pose_measurements:

        # make sure we're tracking a single odometry chain
        if to_pose_key != "":
            assert to_pose_key == odom_measure.from_pose_key

        # update the current pose
        curr_pose = curr_pose @ odom_measure.transformation_matrix
        curr_trans = curr_pose[:-1, -1]

        # initialize the translation variables
        to_pose_key = odom_measure.to_pose_key
        init_translation_variable(model, translations[to_pose_key], curr_trans)


def set_translation_init_random(
    model: MathematicalProgram, translations: Dict[str, np.ndarray]
):
    """initializes the translations to random values

    Args:
        model (MathematicalProgram): the model to initialize the variables in
        translations (Dict[str, np.ndarray]): the translation variables to initialize
    """
    print("Setting translation initial points to random")
    for pose_key in translations:
        init_translation_variable(
            model, translations[pose_key], get_random_vector(dim=2)
        )


def set_landmark_init_gt(
    model: MathematicalProgram,
    landmarks: Dict[str, np.ndarray],
    data: FactorGraphData,
):
    """Initialize the landmark variables to the ground truth landmarks.

    Args:
        landmarks (Dict[str, np.ndarray]): the landmark variables to initialize
        data (FactorGraphData): the factor graph data to use to initialize the landmarks
    """
    print("Setting landmark initial points to ground truth")
    for true_landmark in data.landmark_variables:

        # get landmark position
        landmark_key = true_landmark.name
        landmark_var = landmarks[landmark_key]
        true_pos = np.asarray(true_landmark.true_position)

        # initialize landmark to correct position
        init_translation_variable(model, landmark_var, true_pos)


def set_landmark_init_random(
    model: MathematicalProgram,
    landmarks: Dict[str, np.ndarray],
):
    """Initialize the landmark variables to the ground truth landmarks.

    Args:
        model (MathematicalProgram): the model to initialize the variables in
        landmarks (Dict[str, np.ndarray]): the landmark variables to initialize
    """
    print("Setting landmark initial points to ground truth")
    for landmark in landmarks.values():
        rand_vec = get_random_vector(landmark.shape[0])
        init_translation_variable(model, landmark, rand_vec)


def set_rotation_init_custom(
    model: MathematicalProgram,
    rotations: Dict[str, np.ndarray],
    custom_rotations: Dict[str, np.ndarray],
) -> None:
    """[summary]

    Args:
        model (MathematicalProgram): [description]
        rotations (Dict[str, np.ndarray]): [description]
        custom_rotations (Dict[str, np.ndarray]): [description]
    """
    print("Setting rotation initial points to custom")
    for pose_key in rotations:
        rot = float(custom_rotations[pose_key])
        custom_rot_mat = get_rotation_matrix_from_theta(rot)
        init_rotation_variable(model, rotations[pose_key], custom_rot_mat)


def set_translation_init_custom(
    model: MathematicalProgram,
    translations: Dict[str, np.ndarray],
    custom_translations: Dict[str, np.ndarray],
) -> None:
    """[summary]

    Args:
        model (MathematicalProgram): [description]
        translations (Dict[str, np.ndarray]): [description]
        custom_translations (Dict[str, np.ndarray]): [description]
    """
    print("Setting translation initial points to custom")
    for pose_key in translations:
        init_translation_variable(
            model, translations[pose_key], custom_translations[pose_key]
        )


def set_landmark_init_custom(
    model: MathematicalProgram,
    landmarks: Dict[str, np.ndarray],
    custom_landmarks: Dict[str, np.ndarray],
) -> None:
    """[summary]

    Args:
        model (MathematicalProgram): [description]
        landmarks (Dict[str, np.ndarray]): [description]
        custom_landmarks (Dict[str, np.ndarray]): [description]
    """
    print("Setting landmark initial points to custom")
    for landmark_key in landmarks.keys():
        landmark = landmarks[landmark_key]
        init_translation_variable(model, landmark, custom_landmarks[landmark_key])


def set_distance_init_valid(
    model: MathematicalProgram,
    distances: Dict[Tuple[str, str], np.ndarray],
    custom_translations: Dict[str, np.ndarray],
    custom_landmarks: Dict[str, np.ndarray],
) -> None:
    """[summary]

    Args:
        model (MathematicalProgram): [description]
        distances (Dict[str, np.ndarray]): [description]
        custom_distances (Dict[str, np.ndarray]): [description]
    """
    print("Setting distance initial points to valid based on poses and landmarks")
    for dist_key in distances.keys():
        pose_trans = custom_translations[dist_key[0]]
        landmark = custom_landmarks[dist_key[1]]
        assert pose_trans.shape == landmark.shape

        dx = pose_trans[0] - landmark[0]
        dy = pose_trans[1] - landmark[1]
        dist = np.asarray([(dx ** 2 + dy ** 2) ** (1 / 2)])
        model.SetInitialGuess(distances[dist_key], dist)


##### Constraints #####


def pin_first_pose(
    model: MathematicalProgram,
    translation: np.ndarray,
    rotation: np.ndarray,
    data: FactorGraphData,
    robot_idx: int,
) -> None:
    """
    Pin the first pose of the robot to the origin.

    Args:
        model (MathematicalProgram): The model to pin the pose in
        translation (np.ndarray): The translation variable to pin
        rotation (np.ndarray): The rotation variable to pin
        data (FactorGraphData): The factor graph data to use to pin the pose
        robot_idx (int): The robot index to pin the pose of
    """
    _check_square(rotation)
    assert len(translation) == rotation.shape[0]
    assert 0 <= robot_idx < len(data.pose_variables)

    # fix translation to true position
    true_position = np.asarray(data.pose_variables[robot_idx][0].true_position)
    add_drake_matrix_equality_constraint(model, translation, true_position)

    # fix rotation to identity
    true_rotation = data.pose_variables[robot_idx][0].rotation_matrix
    add_drake_matrix_equality_constraint(model, rotation, true_rotation)


def pin_first_landmark(
    model: MathematicalProgram,
    landmark_position: np.ndarray,
    data: FactorGraphData,
) -> None:
    """
    Pin the first pose of the robot to the origin.

    Args:
        model (MathematicalProgram): The model to pin the pose in
        landmark_position (np.ndarray): The translation variable to pin
        rotation (np.ndarray): The rotation variable to pin
        data (FactorGraphData): The factor graph data to use to pin the pose
        robot_idx (int): The robot index to pin the pose of
    """
    # fix landmark_position to true position
    true_position = np.asarray(data.landmark_variables[0].true_position)
    add_drake_matrix_equality_constraint(model, landmark_position, true_position)


def set_orthogonal_constraint(model: MathematicalProgram, mat: np.ndarray) -> None:
    """Sets an orthogonal constraint on a given matrix (i.e. R.T @ R == I)

    Args:
        model (MathematicalProgram): the model to set the constraint in
        mat (np.ndarray): the matrix to constrain
    """
    assert mat.shape[0] == mat.shape[1], "matrix must be square"
    assert mat.shape[0] == 2, "only support 2d matrices right now"
    d = mat.shape[0]

    for i in range(d):
        for j in range(i, d):
            col_i = mat[:, i]
            if i == j:
                # set diagonal constraint
                const = model.AddConstraint(col_i[0] ** 2 + col_i[1] ** 2 == 1)
            else:
                # set off-diagonal constraint
                col_j = mat[:, j]
                const = model.AddConstraint(
                    col_i[0] * col_j[0] + col_i[1] * col_j[1] == 0
                )


def set_mixed_int_rotation_constraint(
    model: MathematicalProgram, vars: List[np.ndarray]
):
    """Uses built-in Drake constraints to write the rotation matrix as
    mixed-integer constraints

    Args:
        model (MathematicalProgram): the model to set the constraints in
        vars (List[np.ndarray]): the variables to constrain to rotation matrices
    """
    # * various constraints for rotation matrices
    approach = MIRCGenerator.Approach.kBoxSphereIntersection
    # approach = MIRCGenerator.Approach.kBilinearMcCormick
    # approach = MIRCGenerator.Approach.kBoth

    intervals_per_half_axis = 10

    # binning = MIRCGenerator.Binning.kLinear
    binning = MIRCGenerator.Binning.kLogarithmic

    rot_const_generator = MIRCGenerator(approach, intervals_per_half_axis, binning)

    for mat in vars:
        # this is one type of constraint that Drake allows, but is a
        # mixed-integer linear constraint so may be more efficient approaches
        rot_const_generator.AddToProgram(mat, model)


def add_drake_matrix_equality_constraint(
    model: MathematicalProgram, var: np.ndarray, mat: np.ndarray
) -> None:
    """Adds a Drake matrix equality constraint to the model.

    Args:
        model (MathematicalProgram): the model to add the constraint to
        var (np.ndarray): the variable to constrain
        mat (np.ndarray): the matrix to set it equal to
    """
    assert var.shape == mat.shape, "variable and matrix must have same shape"

    var_vec = var.flatten()
    mat_vec = mat.flatten()
    I = np.eye(len(var_vec))
    model.AddLinearEqualityConstraint(
        I,
        mat_vec,
        var_vec,
    )


def add_drake_distance_equality_constraint(
    model: MathematicalProgram, trans: np.ndarray, land: np.ndarray, dist: np.ndarray
) -> None:
    """Adds a constraint of the form
    ||trans-land||_2^2 == dist**2

    Args:
        model (MathematicalProgram): [description]
        trans (np.ndarray): [description]
        land (np.ndarray): [description]
        dist (np.ndarray): [description]
    """
    assert len(trans) == len(land), "must be same dimension"
    assert len(dist) == 1, "must be a scalar"

    d = len(trans)
    I_d = np.eye(len(trans))
    q_size = 2 * d + 1
    Q = np.zeros((q_size, q_size))
    Q[:2, :2] = I_d
    Q[2:4, 2:4] = I_d
    Q[:2, 2:4] = -I_d
    Q[2:4, :2] = -I_d
    Q[-1, -1] = -1

    quad_constraint = QuadraticConstraint(Q, np.zeros((q_size, 1)), lb=0.0, ub=0.0)
    model.AddConstraint(
        quad_constraint,
        vars=np.asarray([*trans.flatten(), *land.flatten(), dist[0]]).flatten(),
    )


##### Misc


def get_solved_values(
    result: DrakeResult,
    time: float,
    translations: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    landmarks: Dict[str, np.ndarray],
    distances: Dict[Tuple[str, str], np.ndarray],
    pose_chain_names: List[str],
) -> SolverResults:
    """
    Returns the solved values from the result

    Args:
        result (Drake Result Object): the result of the solution
        translations (Dict[str, np.ndarray]): the translation variables
        rotations (Dict[str, np.ndarray]): the rotation variables
        landmarks (Dict[str, np.ndarray]): the landmark variables

    Returns:
        Dict[str, np.ndarray]: the solved translations
        Dict[str, np.ndarray]: the solved rotations
        Dict[str, np.ndarray]: the solved landmarks
        Dict[str, np.ndarray]: the solved distances
    """
    solved_translations = {
        key: result.GetSolution(translations[key]) for key in translations.keys()
    }
    solved_rotations = {
        key: np.asarray(round_to_special_orthogonal(result.GetSolution(rotations[key])))
        for key in rotations.keys()
    }
    solved_poses = {
        key: make_transformation_matrix(solved_rotations[key], solved_translations[key])
        for key in translations.keys()
    }
    solved_landmarks = {
        key: result.GetSolution(landmarks[key]) for key in landmarks.keys()
    }
    solved_distances = {
        key: result.GetSolution(distances[key]) for key in distances.keys()
    }

    return SolverResults(
        VariableValues(
            poses=solved_poses,
            landmarks=solved_landmarks,
            distances=solved_distances,
        ),
        total_time=time,
        solved=result.is_success(),
        pose_chain_names=pose_chain_names,
    )


def get_drake_solver(
    solver_name: str,
) -> Union[IpoptSolver, SnoptSolver, GurobiSolver, MosekSolver]:
    """
    Returns the solver for the given name
    """
    if solver_name == "ipopt":
        return IpoptSolver()
    elif solver_name == "snopt":
        return SnoptSolver()
    elif solver_name == "gurobi":
        return GurobiSolver()
    elif solver_name == "mosek":
        return MosekSolver()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def set_drake_solver_verbose(
    model: MathematicalProgram,
    solver: Union[IpoptSolver, SnoptSolver, GurobiSolver, MosekSolver],
):
    """Sets the given solver to verbose output

    Args:
        solver (Union[IpoptSolver, SnoptSolver, GurobiSolver, MosekSolver]): [description]
    """
    if isinstance(solver, IpoptSolver):
        model.SetSolverOption(solver.solver_id(), "print_level", 5)
    elif isinstance(solver, SnoptSolver):
        print("SnoptSolver verbose logging not implemented")
    elif isinstance(solver, GurobiSolver):
        model.SetSolverOption(solver.solver_id(), "OutputFlag", True)
        model.SetSolverOption(solver.solver_id(), "LogToConsole", True)
    elif isinstance(solver, MosekSolver):
        solver.set_stream_logging(True, "/home/alan/mosek_log.out")
    else:
        raise ValueError("Unknown solver")
