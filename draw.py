import cv2
from interval import interval
from environment import ndInterval


def draw_partition(P, image, n, x_list, y_list, exit=0):

    if exit == 0:
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 0)
        partition_color = (70, 242, 70)
        thickness = 5
    else:
        text_font = font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0, 0, 0)
        partition_color = (0, 0, 255)
        thickness = -1

    dimensions = image.shape
    dimensions_maze = (dimensions[0] - 300, dimensions[1] - 310)

    start_point = (int(dimensions_maze[0] * P.inf[0] / n) + 250, dimensions[1] - int(dimensions_maze[1] * P.inf[1] / n) - 250)
    end_point = (start_point[0] + int(dimensions_maze[0] * (P.sup[0] - P.inf[0]) / n), start_point[1] - int(dimensions_maze[1] * (P.sup[1] - P.inf[1]) / n))

    plot = cv2.rectangle(image, start_point, end_point, partition_color, thickness)

    if P.sup[1] not in y_list:
        plot = cv2.putText(plot, str(P.sup[1]), (50, end_point[1] + 15), text_font, 1, text_color, thickness=3)
        y_list.append(P.sup[1])
    if P.inf[1] not in y_list:
        plot = cv2.putText(plot, str(P.inf[1]), (50, start_point[1] + 15), text_font, 1, text_color, thickness=3)
        y_list.append(P.inf[1])
    if P.sup[0] not in x_list:
        plot = cv2.putText(plot, str(P.sup[0]), (end_point[0] - 50, dimensions[1] - 150), text_font, 1, text_color, thickness=3)
        x_list.append(P.sup[0])
    if P.inf[0] not in x_list:
        plot = cv2.putText(plot, str(P.inf[0]), (start_point[0] - 50, dimensions[1] - 150), text_font, 1, text_color, thickness=3)
        x_list.append(P.inf[0])

    return plot, x_list, y_list


def representation(G, env, challenge='Labyrinth'):
    n = env.n
    if challenge == 'U-shape':
        path = './figs/image.png'
    elif challenge == '4-Rooms':
        path = './figs/image4.png'
    elif challenge == 'Labyrinth':
        path = './figs/Labyrinth.png'
    elif challenge == 'N-shape':
        path = './figs/image6.png'
    image = cv2.imread(path)
    x_list = []
    y_list = []
    for i in range(len(G)):
        image, x_list, y_list = draw_partition(G[i], image, n, x_list, y_list)

    P = ndInterval(2, inf=[max(env.exit[0] - 0.3, 0), max(env.exit[1] - 0.3, 0)], sup=[min(env.exit[0] + 0.3, env.n), min(env.exit[1] + 0.3, env.n)])
    image, x_list, y_list = draw_partition(P, image, n, x_list, y_list, exit =1)
    cv2.imshow('partitions', image);
    cv2.waitKey(0);
    cv2.destroyAllWindows()
