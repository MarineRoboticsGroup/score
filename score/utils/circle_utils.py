from typing import Tuple, Optional, List, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import attr
from scipy.spatial import ConvexHull


@attr.s(frozen=True)
class Point:
    x: float = attr.ib()
    y: float = attr.ib()

    @property
    def theta(self):
        angle = np.arctan2(self.y, self.x) + (2 * np.pi)
        return angle % (2 * np.pi)

    @property
    def bearing(self):
        angle = np.arctan2(self.y, self.x) + (2 * np.pi)
        return angle % (2 * np.pi)

    @property
    def distance(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def is_close(self, other: "Point", tol: float = 0.01) -> bool:
        return abs(self.x - other.x) < tol and abs(self.y - other.y) < tol

    def angle_to_point(self, other: "Point") -> float:
        angle = np.arctan2(other.y - self.y, other.x - self.x) + (2 * np.pi)
        return angle % (2 * np.pi)

    def draw_point(self, ax):
        ax.plot(self.x, self.y, "k.")

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def __str__(self):
        return f"Point({self.x:.2f}, {self.y:.2f})"


@attr.s
class Arc:
    """Represents an arc (section of perimeter of circle)"""

    center: Point = attr.ib()
    radius: float = attr.ib()
    thetas: Optional[Tuple[float, float]] = attr.ib()

    @radius.validator
    def check_radius(self, attribute, value):
        assert value > 0, "Radius must be greater than 0"

    @thetas.validator
    def check_thetas(self, attribute, value):
        if value is None:
            return

        assert value[0] <= value[1], "Thetas must be increasing"
        assert all(isinstance(f, float) for f in value), "Thetas must be floats"
        assert all(0 <= x for x in value), "Thetas must be positive"

    def thetas_intersection(self, other: "Arc") -> Optional[Tuple[float, float]]:
        """
        Returns the intersection of the two arcs. This is not a set intersection
        but rather the intersection of the thetas.

        Args:
            other (Arc): The other arc to find the intersection of

        Returns:
            Optional[Tuple[float, float]]: The thetas of the intersection
        """
        assert self.center == other.center
        assert self.radius == other.radius

        if self.is_empty or other.is_empty:
            return None

        two_pi = 2 * np.pi

        # normalize everything by 2pi
        self_thetas = [x % two_pi for x in self.thetas]  # type: ignore
        other_thetas = [x % two_pi for x in other.thetas]  # type: ignore

        # we think about the relative arrangement of the arcs instead of trying
        # to reason about "self" or "other". Without losing generality we
        # arrange them so that whichever arc starts with the least theta is the
        # "origin" or reference arc. By definition the start end point of this
        # arc cannot be in the intersection of the two arcs
        base_range = min([self_thetas, other_thetas], key=lambda x: x[0])
        other_range = max([self_thetas, other_thetas], key=lambda x: x[0])

        base_start, base_end = base_range
        other_start, other_end = other_range

        # print(base_range)
        # print(other_range)

        # some quick rearranging so that the ends are always greater than the
        # starts
        if base_end < base_start:
            base_end += two_pi
        if other_end < other_start:
            other_end += two_pi

        # quick sanity check
        assert base_start <= base_end
        assert other_start <= other_end

        # these distances from the "origin point" (other_start) are all we need
        # to reason about what is going on
        dist_to_other_start = (other_start - base_start) % two_pi
        dist_to_base_end = (base_end - base_start) % two_pi
        dist_to_other_end = (other_end - base_start) % two_pi

        # print("dist to other start", dist_to_other_start)
        # print("dist to base end", dist_to_base_end)
        # print("dist to other end", dist_to_other_end)
        # print()

        # if the distance to base end is closer than other start then either
        # base is fully encapsulated by other or there is no intersection
        if dist_to_base_end < dist_to_other_start:

            if dist_to_other_end < dist_to_base_end:
                int_end = other_end % two_pi
                if int_end < base_start:
                    int_end += two_pi
                pts = (base_start, int_end)
                return pts

            # if other start is closer to base end than to other end then base
            # is fully encapsulated
            dist_other_start_to_base_end = (base_end - other_start) % two_pi
            dist_other_start_to_other_end = (other_end - other_start) % two_pi
            if dist_other_start_to_base_end < dist_other_start_to_other_end:
                return (base_start, base_end)

            return None

        # if the other end is closer than the other start then the intersection
        # is the base start and the other end
        if dist_to_other_end < dist_to_other_start:
            # do some wraparound quick checking to make sure we take the right
            # arc
            int_end = other_end % two_pi
            if int_end < base_start:
                int_end += two_pi

            return (base_start, int_end)

        # otherwise whichever end is closer to the relative start is the end of
        # the intersection
        if dist_to_base_end < dist_to_other_end:
            return (other_start, base_end)
        else:
            # if we have made it here then the other arc must be encapsulated by
            # the base arc and thus the length must be <= the base arc's length
            assert other_start - other_end <= base_end - base_start
            return (other_start, other_end)

    def __str__(self) -> str:
        status = "Arc: "
        status += f"center: {self.center}, "
        status += f"radius: {self.radius}, "
        status += f"thetas: {self.thetas}, "
        status += f"endpoints: {self.end_points}"
        return status

    @property
    def end_points(self):
        if self.is_empty:
            return []

        points = [
            Point(
                self.radius * np.cos(theta) + self.x,
                self.radius * np.sin(theta) + self.y,
            )
            for theta in self.thetas
        ]
        return points

    @property
    def x(self):
        return self.center.x

    @property
    def y(self):
        return self.center.y

    @property
    def arc_length_radians(self):
        """Returns the length of the arc, in radians"""
        if self.is_empty:
            return 0

        diff = self.thetas[1] - self.thetas[0]

        assert diff >= 0, f"Thetas must be increasing: {self.thetas}"
        return diff

    def set_empty(self):
        self.thetas = None

    def set_thetas(self, thetas: Optional[Tuple[float, float]]):
        if thetas is None:
            self.set_empty()
        else:
            assert thetas[0] <= thetas[1], "Thetas must be increasing"
            self.thetas = thetas

    @property
    def is_empty(self) -> bool:
        return self.thetas is None

    def update_with_arc_intersection(self, other: "Arc"):
        """
        Modifies this object in place to represent the intersection of this arc
        with another arc. Each arc must be a part of the same circle.

        Args:
            other (Arc): The other arc to find the intersection of
        """
        assert self.center == other.center
        assert self.radius == other.radius
        assert self.arc_length_radians <= 2 * np.pi
        assert other.arc_length_radians <= 2 * np.pi

        start_len = self.arc_length_radians

        # if thetas is None this arc is already empty
        if self.is_empty:
            return

        # if this arc is a full circle then the intersection is just the other
        # arc
        if abs(self.arc_length_radians - 2 * np.pi) < 1e-3:
            self.set_thetas(other.thetas)
            return

        # if the other arc is a full circle then the intersection is just this
        # arc
        if abs(other.arc_length_radians - 2 * np.pi) < 1e-3:
            return

        intersection = self.thetas_intersection(other)
        if intersection is None:
            # print(f"No intersection between arcs - setting empty")
            # print(self)
            # print(other)
            self.set_empty()
        else:
            self.set_thetas(intersection)

        new_len = self.arc_length_radians
        assert new_len <= start_len, f"New arc length must be <= old length"

    def draw_arc_patch(
        self,
        ax: plt.Axes,
        resolution: int = 50,
        color: str = "blue",
    ) -> Optional[patches.Polygon]:
        """Draws an arc as a generic patches.Polygon

        Args:
            arc (Arc): the arc to draw
            ax (plt.Axes): the axes to draw the arc on
            resolution (int, optional): the resolution of the arc. Defaults to
            50.
            color (str, optional): the color of the arc. Defaults to "black".

        Returns:
            patches.Polygon: the arc
        """
        if self.is_empty:
            return None

        radius = self.radius
        theta1, theta2 = self.thetas  # type: ignore
        if theta2 < theta1:
            theta2 += 2 * np.pi
        # generate the points
        theta = np.linspace((theta1), (theta2), resolution)
        points = np.vstack(
            (radius * np.cos(theta) + self.x, radius * np.sin(theta) + self.y)
        )
        # build the polygon and add it to the axes
        coloring = to_rgba(color, 0.5)
        poly = patches.Polygon(points.T, closed=True, color=coloring, linewidth=0)
        ax.add_patch(poly)
        return poly


@attr.s
class Circle:
    """A circle object

    Attributes:
        x (float): x coordinate of the center of the circle
        y (float): y coordinate of the center of the circle
        radius (float): radius of the circle
    """

    center: Point = attr.ib()
    radius: float = attr.ib()

    @radius.validator
    def check_radius(self, attribute, value):
        assert value > 0, "Radius must be greater than 0"

    @property
    def x(self):
        return self.center.x

    @property
    def y(self):
        return self.center.y

    def angle_to_point(self, point: Point) -> float:
        """returns the angle from the center of the circle to a given point

        Args:
            point (Point): [description]

        Returns:
            float: [description]
        """
        return self.center.angle_to_point(point)

    def is_inside_circle(self, point: Point) -> bool:
        """
        Returns true if the given point is inside the circle

        Args:
            point (Point): The point to check

        Returns:
            bool: True if the point is inside the circle
        """
        return (point - self.center).distance <= self.radius

    def completely_contains(self, other: "Circle") -> bool:
        """
        Returns true if the given circle is completely inside "self"

        Args:
            other (Circle): The other circle to check

        Returns:
            bool: True if other is completely inside "self"
        """

        # the other circle is not inside this circle
        other_is_inside_self = self.is_inside_circle(other.center)
        if not other_is_inside_self:
            return False

        # the other circle is inside this circle, but not completely
        circ_dist = (other.center - self.center).distance
        if circ_dist + other.radius > self.radius:
            return False

        return True

    def get_intersection_point_angles(
        self, other: "Circle"
    ) -> Optional[Tuple[float, float]]:
        """[summary]

        Args:
            other (Circle): [description]

        Returns:
            Optional[Tuple[float, float]]: [description]
        """
        d = np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

        if d > self.radius + other.radius:
            print("Circles too far away")
            return None

        min_rad = min(self.radius, other.radius)
        max_rad = max(self.radius, other.radius)
        if d + min_rad < max_rad:
            print("Circles nested inside eachother")
            return None

        l = (self.radius ** 2 - other.radius ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(self.radius ** 2 - l ** 2)

        x_int_base = self.x + l * (other.x - self.x) / d
        x_int_1 = x_int_base + h * (other.y - self.y) / d
        x_int_2 = x_int_base - h * (other.y - self.y) / d

        y_int_base = self.y + l * (other.y - self.y) / d
        y_int_1 = y_int_base - h * (other.x - self.x) / d
        y_int_2 = y_int_base + h * (other.x - self.x) / d

        int_pt_1 = Point(x_int_1, y_int_1)
        int_pt_2 = Point(x_int_2, y_int_2)
        pt_list = [int_pt_1, int_pt_2]

        centered_pts = [pt - self.center for pt in pt_list]

        assert all(abs(pt.distance - self.radius) < 1e-3 for pt in centered_pts)

        pt_thetas = [pt.theta for pt in centered_pts]
        pt_thetas.sort()
        assert pt_thetas[0] < pt_thetas[1]
        return (float(pt_thetas[0]), float(pt_thetas[1]))

    def get_circle_intersection_arc(self, other: "Circle") -> Optional[Arc]:
        """
        Returns the intersection of two circles

        Args:
            other (Circle): The other circle to find the intersection of

        Returns:
            Optional[Arc]: The arc on this circle representing the boundary of
            the intersection of the circles
        """

        # if the other circle is completely inside this circle none of this
        # circle makes up the boundary
        other_is_completely_inside_self = self.completely_contains(other)
        if other_is_completely_inside_self:
            print("other is completely inside self")
            return None

        # if this circle is completely inside the other circle then the entire
        # circle makes up the boundary
        self_is_completely_inside_other = other.completely_contains(self)
        if self_is_completely_inside_other:
            print("self is completely inside other")
            return Arc(self.center, self.radius, (0.0, 2 * np.pi))

        # if there is no intersection of the circles and neither is completely
        # inside the other then return None
        intersect_is_null = circles_have_no_overlap(self, other)
        if intersect_is_null:
            print("Circles have no intersection")
            return None

        # we've now checked the reasons why there might be no intersection, so
        # we get the intersection points
        intersect_point_thetas = self.get_intersection_point_angles(other)
        assert (
            intersect_point_thetas is not None
        ), "Intersection points should not be None - have already weeded out possible causes of that"

        # these are the angles to the intersection points - sorted least to
        # greatest
        angle_to_pt_1, angle_to_pt_2 = intersect_point_thetas

        # make two arc objects and we will choose between the larger or smaller
        # one as described below
        arc1 = Arc(self.center, self.radius, (angle_to_pt_1, angle_to_pt_2))
        arc2 = Arc(
            self.center, self.radius, (angle_to_pt_2, angle_to_pt_1 + (2 * np.pi))
        )
        larger_arc = max(arc1, arc2, key=lambda arc: arc.arc_length_radians)
        smaller_arc = min(arc1, arc2, key=lambda arc: arc.arc_length_radians)

        # from here based on the relative angles between the centers of the
        # circles and the intersection points, we can determine whether to use
        # the larger or smaller arc. We get both the relative angles to avoid
        # having to think about 2pi wraparound issues
        angle_to_other_center = self.angle_to_point(other.center)
        rel_angle_1 = abs(angle_to_other_center - angle_to_pt_1)
        rel_angle_2 = abs(angle_to_other_center - angle_to_pt_2)
        relative_angle = min(rel_angle_1, rel_angle_2)
        if (relative_angle) > np.pi / 2:
            return larger_arc
        else:
            return smaller_arc

    def draw_ray(self, theta: float, ax: plt.Axes) -> None:
        """
        Draws a ray from the center of this circle to a point on the circle
        that is the given angle away from the center

        Args:
            theta (float): The angle of the ray to draw
        """
        dist = 10
        ray_end_x = self.x + dist * np.cos(theta)
        ray_end_y = self.y + dist * np.sin(theta)
        ax.plot([self.x, ray_end_x], [self.y, ray_end_y], color="black")

    def draw_circle_patch(
        self,
        ax: plt.Axes,
        color: str = "black",
    ) -> patches.Circle:
        """Draws a circle as a generic patches.Circle

        Args:
            circle (Circle): the circle to draw
            ax (plt.Axes): the axes to draw the circle on
            resolution (int, optional): the resolution of the circle. Defaults to
            50.
            color (str, optional): the color of the circle. Defaults to "black".

        Returns:
            patches.Circle: the circle
        """
        circle = patches.Circle((self.x, self.y), self.radius, fill=False, color=color)
        ax.add_patch(circle)
        return circle


@attr.s
class CircleIntersection:
    """
    Represents the intersection of generic number of circles

    Attributes:
        circles (List[Circle]): The circles that make up the intersection
        arcs (List[Arc]): The arcs that make up the boundary of the intersection (these are what are actually drawn on the plot)
        intersect_is_null (bool): this is a flag for when we've decided that the intersection must be empty from here onwards
    """

    circles: List[Circle] = attr.ib(default=attr.Factory(list))
    intersection_arcs: List[Arc] = attr.ib(default=attr.Factory(list))
    intersect_is_null: bool = attr.ib(default=False)

    def add_circle(self, new_circle: Circle) -> None:
        """
        Adds a circle to the intersection

        Args:
            new_circle (Circle): the circle to add
        """
        self.circles.append(new_circle)

        # if the intersection has been decided to be null then we won't add any
        # more arcs
        if self.intersect_is_null:
            empty_arc = Arc(new_circle.center, new_circle.radius, None)
            self.intersection_arcs.append(empty_arc)
            return

        # this should only be entered if this is the first circle added to the
        # intersection so we can just add it and exit
        if len(self.circles) == 1:
            first_arc = Arc(new_circle.center, new_circle.radius, (0.0, 2 * np.pi))
            self.intersection_arcs.append(first_arc)
            return

        # from here onwards this arc will be iteratively trimmed as we consider
        # all of the existing arcs to see if they intersect with the new arc
        new_intersection_arc = Arc(
            new_circle.center, new_circle.radius, (0.0, 2 * np.pi)
        )

        assert len(self.circles[:-1]) == len(
            self.intersection_arcs
        ), "The number of circles and arcs should be the same"
        for existing_circ, existing_arc in zip(
            self.circles[:-1], self.intersection_arcs
        ):

            # if any of the existing circles have no intersection with the new
            # circle then the intersection is null from here onwards
            if circles_have_no_overlap(new_circle, existing_circ):
                print("No intersection in circles - setting null")
                empty_arc = Arc(new_circle.center, new_circle.radius, None)
                self.intersection_arcs.append(empty_arc)
                for arc in self.intersection_arcs:
                    arc.set_empty()
                self.intersect_is_null = True
                return

            # if any of the existing circles are completely inside of the new
            # then we won't add it to the intersection list and we exit because
            # the intersection is already a subset of this circle
            if new_circle.completely_contains(existing_circ):
                empty_arc = Arc(new_circle.center, new_circle.radius, None)
                self.intersection_arcs.append(empty_arc)
                return

            # Note: if the existing arc is already empty then this existing
            # circle has no impact on that arc but it can still impact this new
            # arc (this was causing a bug for me!)

            # for every circle this circle is completely inside of we can
            # eliminate that circle and its arc
            if existing_circ.completely_contains(new_circle):
                # print("Completely contains - setting empty")
                existing_arc.set_empty()
                continue

            # if we've made it this far then they must have an intersection
            new_intersect = new_circle.get_circle_intersection_arc(existing_circ)
            assert new_intersect is not None, "new_intersect should not be None"
            new_intersection_arc.update_with_arc_intersection(new_intersect)

            exist_intersect = existing_circ.get_circle_intersection_arc(new_circle)
            assert exist_intersect is not None, "exist_intersect should not be None"
            existing_arc.update_with_arc_intersection(exist_intersect)

        self.intersection_arcs.append(new_intersection_arc)

        assert all(x is not None for x in self.circles), "Circles should not be None"
        assert all(
            x is not None for x in self.intersection_arcs
        ), f"Intersection arcs should not be None: {self.intersection_arcs}"

    def draw_circles(self, ax: plt.Axes, color: str = "red") -> None:
        """
        Draws all of the circles in the intersection

        Args:
            ax (plt.Axes): the axes to draw the circles on
            color (str, optional): the color of the circles. Defaults to "red".
        """
        for circle in self.circles[-3:]:
            circle.draw_circle_patch(ax, color=color)

    def draw_intersection(self, ax: plt.Axes, color: str = "blue") -> None:
        """
        Draws the intersection of all of the circles

        Args:
            ax (plt.Axes): the axes to draw the intersection on
            color (str, optional): the color of the intersection. Defaults to
            "blue".
        """
        if self.intersect_is_null:
            return

        fill_points: Set[Point] = set()
        for arc in self.intersection_arcs:
            if not arc.is_empty:

                # get the endpoints for filling in the intersection not captured
                # by the arcs
                endpt1, endpt2 = arc.end_points
                if all(not pt.is_close(endpt1) for pt in fill_points):
                    fill_points.add(endpt1)
                if all(not pt.is_close(endpt2) for pt in fill_points):
                    fill_points.add(endpt2)

                # draw the arc
                arc.draw_arc_patch(ax, color=color)

        # if the intersection not captured by the arcs is more than a line
        # segment then we make the polygon and fill it in
        if len(fill_points) > 2:
            xy_pts = []
            for pt in fill_points:
                xy_pts.append([pt.x, pt.y])

            hull = ConvexHull(np.asarray(xy_pts))
            hull_pts = np.asarray(xy_pts)[hull.vertices]

            coloring = to_rgba(color, 0.5)
            fill_poly = patches.Polygon(hull_pts, True, fc=coloring, linewidth=0)
            ax.add_patch(fill_poly)


def circles_have_no_overlap(c1: Circle, c2: Circle) -> bool:
    """
    Returns true if the given circle has no overlap with "self"

    Args:
        c1: The first circle
        c2: The second circle

    Returns:
        bool: True if the circles have no overlap

    """
    circle_dist = (c1.center - c2.center).distance
    circles_have_no_overlap = circle_dist > c1.radius + c2.radius
    return circles_have_no_overlap


def get_random_circle(
    max_x: float,
    max_y: float,
    max_radius: float,
) -> Circle:
    """
    Returns a random circle with the given radius

    Args:
        max_x (float): The maximum x value of the circle
        max_y (float): The maximum y value of the circle
        max_radius (float): The radius of the circle

    Returns:
        Circle: A random circle
    """
    x = np.random.uniform(-max_x, max_x)
    y = np.random.uniform(-max_y, max_y)
    center = Point(x, y)
    rad = np.random.uniform(0, max_radius)
    return Circle(center, rad)


if __name__ == "__main__":
    np.random.seed(0)

    t0 = False
    t1 = False
    t2 = True

    if t0:
        a1_angles = (4.889303064250705, 7.622526837716786)
        a2_angles = (7.371161260586642, 10.104385034052722)

        a1 = Arc(Point(0, 0), 1, a1_angles)
        a2 = Arc(Point(0, 0), 1, a2_angles)

        fig, ax = plt.subplots()
        a1.draw_arc_patch(ax, color="red")
        a2.draw_arc_patch(ax, color="blue")

        a1.update_with_arc_intersection(a2)
        a1.draw_arc_patch(ax, color="green")

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        plt.show(block=True)
        plt.show()

    elif t1:

        # fig, ax = plt.subplots()
        n_step = 100
        for i in range(n_step):
            print()
            fig, ax = plt.subplots()

            a1_start = np.random.uniform(0, 2 * np.pi)
            a1_end = a1_start + np.random.uniform(0, np.pi)
            a1_angles = (a1_start, a1_end)
            a1 = Arc(Point(0, 0), 1, a1_angles)

            offset = i * np.pi / (n_step / 6) + np.pi / 4
            a2_angles = (a1_start + offset, a1_end + offset)
            a2 = Arc(Point(0, 0), 1, a2_angles)

            a1.draw_arc_patch(ax, color="red")
            a2.draw_arc_patch(ax, color="blue")

            a1.update_with_arc_intersection(a2)
            a1.draw_arc_patch(ax, color="green")

            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")
            plt.show(block=True)
            # plt.pause(0.2)
            # ax.patches = []

    elif t2:
        num_circles = 5

        fig, ax = plt.subplots()
        for _ in range(100):

            c1 = Circle(Point(0, 0), 1)
            intersect_list = CircleIntersection()
            intersect_list.add_circle(c1)
            max_x = 0.5
            max_y = 0.5
            max_radius = 1

            for _ in range(num_circles - 1):
                rand_circ = get_random_circle(max_x, max_y, max_radius)
                rand_circ.radius = 1
                intersect_list.add_circle(rand_circ)

            # draw what we've found
            intersect_list.draw_intersection(ax)
            # intersect_list.draw_circles(ax)
            # for circ in intersect_list.circles:
            #     print(circ)
            # for arc in intersect_list.intersection_arcs:
            #     print(arc)
            # print()

            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect("equal")
            plt.show(block=False)
            plt.pause(0.3)
            # ax.patches = []
            ax.clear()
