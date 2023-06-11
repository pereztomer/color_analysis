import numpy as np

conversion_matrix = np.array(
    [[-0.0013, 0.9057, 0.1440, -0.0483], [-0.0389, -0.2808, 1.6861, -0.3625], [-0.0389, -0.0549, -0.0861, 1.9578]])


def test_matrix_multyplciation(pixel):
    L = 0
    a = 0
    b = 0
    for index in range(4):
        if index == 3:
            L += conversion_matrix[0][3]
            a += conversion_matrix[1][3]
            b += conversion_matrix[2][3]
        else:
            L += pixel[index] * conversion_matrix[0][index]
            a += pixel[index] * conversion_matrix[1][index]
            b += pixel[index] * conversion_matrix[2][index]

    return L, a, b


def f(val):
    k = 903.3  # 24389 / 27
    epsilon = 0.008856  # 216 / 24389
    if val > epsilon:
        return val ** (1 / 3)
    return (k * val + 16) / 116


def srgb_expand_func(val):
    if val <= 10:
        return val / 3294.6
    return ((val + 14.025) / 269.025) ** 2.4


def main():
    # convert_to_lab([255, 0,0, 1])
    from skimage import io, color
    # rgb = io.imread('/home/tomer/Jubban/full_db_flavor_C/C2460.png')
    rgb = np.array([[[7 / 255, 22 / 255, 56 / 255]]])
    lab = color.rgb2lab(rgb)
    print(lab)
    srgb_dot = np.array([7, 22, 56])
    expanded_srgb = np.array([srgb_expand_func(x) for x in srgb_dot])

    srgb2xyz = np.array([[0.4124530, 0.3575800, 0.180423],
                         [0.212671, 0.715160, 0.072169],
                         [0.019334, 0.119193, 0.950227]])
    xyz = srgb2xyz @ expanded_srgb
    x_func_out = f(xyz[0] / 0.95044921)
    y_func_out = f(xyz[1])
    z_func_out = f(xyz[2] / 1.08888166)
    L = (116 * y_func_out - 16)
    a = 500 * (x_func_out - y_func_out)
    b = 200 * (y_func_out - z_func_out)
    print(f'L*a*b values: L:{L} a:{a} b:{b}')


def convert_to_lab(rgb_dot):
    conversion_matrix = np.array(
        [[-0.0013, 0.9057, 0.1440, -0.0483], [-0.0389, -0.2808, 1.6861, -0.3625], [-0.0389, -0.0549, -0.0861, 1.9578]])
    xyz = conversion_matrix @ rgb_dot
    xyz = xyz / np.sum(xyz)
    print(xyz)
    x_func_out = f(xyz[0] / 0.9304)
    y_func_out = f(xyz[1])
    z_func_out = f(xyz[2] / 1.0888)
    L = 166 * y_func_out - 16
    a = 500 * (x_func_out - y_func_out)
    b = 200 * (y_func_out - z_func_out)
    print(f'L*a*b values: L:{L} a:{a} b:{b}')


if __name__ == '__main__':
    main()
