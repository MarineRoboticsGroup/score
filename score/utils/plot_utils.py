from typing import Dict, Tuple, List, Optional
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import to_rgba
from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.variables import PoseVariable, LandmarkVariable
from ro_slam.utils.circle_utils import Arc, Circle, CircleIntersection, Point
from ro_slam.utils.solver_utils import SolverResults, VariableValues
from ro_slam.utils.matrix_utils import (
    _check_transformation_matrix,
    get_theta_from_rotation_matrix,
    get_translation_from_transformation_matrix,
)

colors = ["red", "green", "blue", "orange", "purple", "black", "cyan"]


def plot_error(
    data: FactorGraphData,
    solved_results: SolverResults,
    initial_values: Optional[VariableValues] = None,
    color_dist_circles: bool = False,
) -> None:
    """
    Plots the error for the given data

    Args:
        data (FactorGraphData): the groundtruth data
        solved_results (Dict[str, Dict[str, np.ndarray]]): the solved values of the variables
        initial_values (VariableValues): the initial values of the variables
        before solving
        color_dist_circles (bool, optional): whether to display the circles
        indicating the distance measurements. Defaults to False.

    """

    def draw_all_information(
        ax: plt.Axes,
        gt_data: FactorGraphData,
        solution_data: SolverResults,
        init_vals: Optional[VariableValues] = None,
        use_arrows: bool = True,
    ):
        """Draws the pose estimates and groundtruth

        Args:
            ax (plt.Axes): the axes to draw on
            gt_data (FactorGraphData): the groundtruth data
            solution_results (SolverResults): the solved values of the variables
        """
        assert gt_data.num_poses > 0
        max_pose_chain_length = max(
            [len(pose_chain) for pose_chain in gt_data.pose_variables]
        )
        num_landmarks = len(gt_data.landmark_variables)

        true_poses_dict = gt_data.pose_variables_dict
        loop_closures = gt_data.loop_closure_measurements
        loop_closure_dict = {
            x.base_pose: true_poses_dict[x.to_pose] for x in loop_closures
        }

        for landmark in gt_data.landmark_variables:
            draw_landmark_variable(ax, landmark)
            draw_landmark_solution(ax, solution_data.landmarks[landmark.name])

        pose_var_plot_obj: List[mpatches.FancyArrow] = []
        pose_sol_plot_obj: List[mpatches.FancyArrow] = []
        pose_init_val_plot_obj: List[mpatches.FancyArrow] = []
        range_circles: List[CircleIntersection] = [
            CircleIntersection() for _ in range(num_landmarks)
        ]
        pose_to_range_measures_dict = gt_data.pose_to_range_measures_dict

        # draw range measurements
        range_measure_plot_lines: List[mlines.Line2D] = []

        cnt = 0
        num_frames_skip = 2
        for pose_idx in range(max_pose_chain_length):
            if cnt % num_frames_skip == 0:
                cnt = 0
            else:
                cnt += 1
                continue

            for pose_chain in gt_data.pose_variables:

                if len(pose_chain) == 0:
                    continue

                # if past end of pose chain just grab last pose, otherwise use
                # next in chain
                if len(pose_chain) <= pose_idx:
                    pose = pose_chain[-1]
                else:
                    pose = pose_chain[pose_idx]

                # draw inferred solution
                soln_arrow = draw_pose_solution(
                    ax,
                    solution_data.poses[pose.name],
                )
                pose_sol_plot_obj.append(soln_arrow)

                if init_vals is not None:
                    # draw the initial point used
                    init_arrow = draw_pose_solution(
                        ax,
                        init_vals.poses[pose.name],
                        color="green",
                        alpha=0.5,
                    )
                    pose_init_val_plot_obj.append(init_arrow)

                if pose.name in pose_to_range_measures_dict:
                    cur_range_measures = pose_to_range_measures_dict[pose.name]

                    for range_measure in cur_range_measures:
                        range_var1, range_var2 = range_measure.association
                        x1, y1 = solution_data.translations[range_var1]
                        if "L" == range_var2[0]:
                            x2, y2 = solution_data.landmarks[range_var2]
                        else:
                            x2, y2 = solution_data.translations[range_var2]

                        new_line = draw_line(ax, x1, y1, x2, y2, color="red")
                        range_measure_plot_lines.append(new_line)

                # draw arc to inferred landmarks
                for landmark_idx, landmark in enumerate(gt_data.landmark_variables):
                    soln_pose_center = solution_data.translations[pose.name]
                    soln_landmark_center = solution_data.landmarks[landmark.name]
                    range_key = (pose.name, landmark.name)
                    if color_dist_circles and range_key in pose_to_range_measures_dict:
                        arc_radius = pose_to_range_measures_dict[range_key]
                        dist_circle = Circle(
                            Point(soln_pose_center[0], soln_pose_center[1]), arc_radius
                        )
                        range_circles[landmark_idx].add_circle(dist_circle)
                    range_circles[landmark_idx].draw_intersection(
                        ax, color=colors[landmark_idx]
                    )
                    # range_circles[landmark_idx].draw_circles(
                    #     ax, color=colors[landmark_idx]
                    # )

                # draw groundtruth solution
                var_arrow = draw_pose_variable(ax, pose)
                pose_var_plot_obj.append(var_arrow)

                # if loop closure draw it
                if pose.name in loop_closure_dict:
                    loop_line, loop_pose = draw_loop_closure_measurement(
                        ax,
                        solution_data.translations[pose.name],
                        loop_closure_dict[pose.name],
                    )
                else:
                    loop_line = None
                    loop_pose = None

            plt.pause(0.001)
            ax.patches.clear()
            # print(ax.patches)
            # ax.patches = []
            # if loop_line and loop_pose:
            #     loop_line.remove()
            #     loop_pose.remove()

            while len(range_measure_plot_lines) > 0:
                range_measure_plot_lines.pop().remove()

            # if pose_idx > 5:
            #     # ax.remove(pose_sol_plot_obj[0])
            #     pose_sol_plot_obj[0].remove()
            #     pose_sol_plot_obj.pop(0)
            #     pose_var_plot_obj[0].remove()
            #     pose_var_plot_obj.pop(0)
            #     if init_vals is not None:
            #         pose_init_val_plot_obj[0].remove()
            #         pose_init_val_plot_obj.pop(0)

        plt.close()

    # set up plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(data.x_min - 1, data.x_max + 1)
    ax.set_ylim(data.y_min - 1, data.y_max + 1)

    # draw all poses to view static image result
    draw_all_information(ax, data, solved_results)


def draw_arrow(
    ax: plt.Axes,
    x: float,
    y: float,
    theta: float,
    color: str = "black",
) -> mpatches.FancyArrow:
    """Draws an arrow on the given axes

    Args:
        ax (plt.Axes): the axes to draw the arrow on
        x (float): the x position of the arrow
        y (float): the y position of the arrow
        theta (float): the angle of the arrow
        quiver_length (float, optional): the length of the arrow. Defaults to 0.1.
        quiver_width (float, optional): the width of the arrow. Defaults to 0.01.
        color (str, optional): color of the arrow. Defaults to "black".

    Returns:
        mpatches.FancyArrow: the arrow
    """
    plot_x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    plot_y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

    quiver_length: float = max(plot_x_range, plot_y_range) / 20
    quiver_width: float = max(plot_x_range, plot_y_range) / 100
    dx = quiver_length * math.cos(theta)
    dy = quiver_length * math.sin(theta)
    return ax.arrow(
        x,
        y,
        dx,
        dy,
        head_width=quiver_length,
        head_length=quiver_length,
        width=quiver_width,
        color=color,
    )


def draw_line(
    ax: plt.Axes,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    color: str = "black",
) -> mlines.Line2D:
    """Draws a line on the given axes between the two points

    Args:
        ax (plt.Axes): the axes to draw the arrow on
        x_start (float): the x position of the start of the line
        y_start (float): the y position of the start of the line
        x_end (float): the x position of the end of the line
        y_end (float): the y position of the end of the line
        color (str, optional): color of the arrow. Defaults to "black".

    Returns:
        mpatches.FancyArrow: the arrow
    """
    line = mlines.Line2D([x_start, x_end], [y_start, y_end], color=color)
    ax.add_line(line)
    return line


def draw_pose_variable(ax: plt.Axes, pose: PoseVariable):
    true_x = pose.true_x
    true_y = pose.true_y
    true_theta = pose.true_theta
    return draw_arrow(ax, true_x, true_y, true_theta, color="blue")


def draw_pose_solution(
    ax: plt.Axes,
    pose: np.ndarray,
    color: str = "red",
    alpha: float = 1.0,
):
    _check_transformation_matrix(pose)
    x, y = get_translation_from_transformation_matrix(pose)
    theta = get_theta_from_rotation_matrix(pose[0:2, 0:2])

    coloring = to_rgba(color, alpha)
    return draw_arrow(ax, x, y, theta, color=coloring)


def draw_loop_closure_measurement(
    ax: plt.Axes, base_loc: np.ndarray, to_pose: PoseVariable
) -> Tuple[mlines.Line2D, mpatches.FancyArrow]:
    assert base_loc.size == 2

    x_start = base_loc[0]
    y_start = base_loc[1]
    x_end = to_pose.true_x
    y_end = to_pose.true_y

    line = draw_line(ax, x_start, y_start, x_end, y_end, color="green")
    arrow = draw_pose_variable(ax, to_pose)

    return line, arrow


def draw_landmark_variable(ax: plt.Axes, landmark: LandmarkVariable):
    true_x = landmark.true_x
    true_y = landmark.true_y
    ax.scatter(true_x, true_y, color="green", marker=(5, 2))


def draw_landmark_solution(ax: plt.Axes, translation: np.ndarray):
    x = translation[0]
    y = translation[1]
    ax.scatter(x, y, color="red", marker=(4, 2))


def draw_arc_patch(
    arc: Arc,
    ax: plt.Axes,
    resolution: int = 50,
    color: str = "black",
) -> mpatches.Polygon:
    """Draws an arc as a generic mpatches.Polygon

    Args:
        arc (Arc): the arc to draw
        ax (plt.Axes): the axes to draw the arc on
        resolution (int, optional): the resolution of the arc. Defaults to
        50.
        color (str, optional): the color of the arc. Defaults to "black".

    Returns:
        mpatches.Polygon: the arc
    """
    center = arc.center
    radius = arc.radius

    assert arc.thetas is not None
    theta1, theta2 = arc.thetas

    # generate the points
    theta = np.linspace((theta1), (theta2), resolution)
    points = np.vstack(
        (radius * np.cos(theta) + center.x, radius * np.sin(theta) + center.y)
    )
    # build the polygon and add it to the axes
    poly = mpatches.Polygon(points.T, closed=True)
    ax.add_patch(poly)
    return poly
