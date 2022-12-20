import numpy as np

BORDER_WIDTH = 10


def gen_h_line(size: int) -> np.array:
    img, x, y, l, w = _gen_img_x_y_l_w(size)
    img[x - w : x + w, y - l : y + l] = 1
    if np.array_equal(img, np.zeros([size, size])):
        return gen_h_line(size)
    return img


def gen_v_line(size: int) -> np.array:
    img, x, y, l, w = _gen_img_x_y_l_w(size)
    img[x - l : x + l, y - w : y + w] = 1
    if np.array_equal(img, np.zeros([size, size])):
        return gen_v_line(size)
    return img


def _gen_img_x_y_l_w(size: int) -> tuple:
    border = BORDER_WIDTH
    img = np.zeros([size, size])
    x = np.random.randint(border, size - border)
    y = np.random.randint(border, size - border)
    l = np.random.randint((size - border) // 8, (size - border) // 2)
    w = np.random.randint(1, 4)

    return img, x, y, l, w
