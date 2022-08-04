from matplotlib import pyplot as plt
from point import Point

import os.path


def parse_points(input: str):
    my_path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(my_path, "./input.txt")

    points = []

    fh = open(path)
    for line in fh:
        p = [l for l in line.split(' ') if l != '']
        points.append(Point(int(p[0]), int(p[1])))

    fh.close()

    return points


def draw(fast, points):
    for line_segment in fast:
        plt.plot([line_segment.p1.x, line_segment.p2.x], [
                 line_segment.p1.y, line_segment.p2.y], linewidth=1)

    x_points, y_points = [p.x for p in points], [p.y for p in points]
    plt.scatter(x_points, y_points, s=10)

    # Set chart title.
    plt.title('Collinear points')

    # Set x, y label text.
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()
