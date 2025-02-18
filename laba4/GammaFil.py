import numpy as np
from PIL import Image
import random
import math
import cv2
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

def factorial(n):
    return math.factorial(n)

def GammaNoise(image, a, b, image_name):
    # f = cv2.imread(image)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = Image.open(image)
    # Получаем размеры изображения
    w, h = image.size
    # Преобразуем изображение в массив NumPy
    img_array = np.array(image)

   
    # Получаем общее количество байт (пикселей * каналы)
    bytes_count = img_array.size

    # Создаем массив для шума
    noise = np.zeros(bytes_count, dtype=np.uint8)

    # Параметры для распределения Эрланга
    a = a
    b = b

    # Вычисляем распределение Эрланга
    erlang = np.zeros(256, dtype=np.float64)
    sum_erlang = 0.0

    for i in range(256):
        step = i * 0.1
        if step >= 0:
            erlang[i] = math.exp(-a * step) * (math.pow(a, b) * math.pow(step, b - 1)) / factorial(int(b) - 1)
        else:
            erlang[i] = 0
        sum_erlang += erlang[i]

    # Нормализуем распределение и умножаем на количество байт
    erlang = (erlang / sum_erlang * bytes_count).astype(int)

    # Заполняем массив шума в соответствии с распределением Эрланга
    count = 0
    for i in range(256):
        noise[count:count + erlang[i]] = i
        count += erlang[i]

    # Дополняем оставшиеся элементы нулями
    noise[count:] = 0

    # Перемешиваем шум
    np.random.shuffle(noise)

    # Преобразуем шум в форму изображения
    noise = noise.reshape(img_array.shape)

    # Добавляем шум к изображению
    result_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    # Преобразуем массив обратно в изображение
    result_image = Image.fromarray(result_array)
    
    result_image.save(image_name)

    img = mpimg.imread(image_name)
    plt.figure(figsize=(8, 6))  
    plt.imshow(img)              
    plt.axis('off')              
    plt.title(type_noise) 
    plt.show()

    
    return result_image
