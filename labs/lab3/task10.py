"""
Задача N10.

Дан трёхмерный массив, содержащий изображение, размера (height, width, numChannels),
а также вектор длины numChannels.

Написать функцию, которая складывает каналы изображения с указанными весами, и возвращает
результат в виде матрицы размера (height, width)
"""
import numpy as np

IMAGE_FILENAME = "image.txt"
CHANNEL_WEIGHTS_FILENAME = "channel_weights.txt"
RESULT_FILENAME = "result.txt"


def main():
    print("============================================")
    print("================ Задача 10. ================")
    print("============================================")

    image = read_image_data_from_file(IMAGE_FILENAME)
    channel_weights = np.loadtxt(CHANNEL_WEIGHTS_FILENAME)

    height, width, num_channels = image.shape

    print(f"{height=}")
    print(f"{width=}")
    print(f"{num_channels=}")

    print(f"{image=}")
    print(f"{channel_weights=}")

    result = np.zeros((height, width))
    for i, pixel in enumerate(image):
        for j, channel in enumerate(pixel):
            result[i, j] = np.sum(channel * channel_weights)

    print(f"{result=}")
    np.savetxt(RESULT_FILENAME, result)


def read_image_data_from_file(filename: str) -> np.array:
    with open(filename, "r") as file:
        height = int(file.readline())
        width = int(file.readline())
        num_channels = int(file.readline())

        image = []
        for h in range(height):
            pixel = []
            for w in range(width):
                weights = np.array(file.readline().split(",")[:num_channels], dtype=float)
                pixel.append(weights)
            image.append(pixel)

        return np.array(image, dtype=float)


if __name__ == "__main__":
    main()
