from utils import draw
from collinear import ColinearPoints
from utils import parse_points

points = parse_points("./input.txt")

cp = ColinearPoints()

fast = cp.collinear_points_fast(points)

draw(fast, points)