import numpy as np
from PIL import Image
import os

def create_striped_image(size=(256, 256)):
    """Создание изображения с горизонтальными полосами и белым кругом"""
    image = np.zeros(size, dtype=np.uint8)
    
    # Горизонтальные полосы
    stripe_height = size[0] // 10
    for i in range(0, size[0], stripe_height * 2):
        image[i:i+stripe_height] = 0  # черная полоса
        if i+stripe_height < size[0]:
            image[i+stripe_height:i+stripe_height*2] = 255  # белая полоса
    
    # Белый круг в центре
    center_y, center_x = size[0] // 2, size[1] // 2
    radius = min(size) // 2.5
    
    y, x = np.ogrid[:size[0], :size[1]]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    image[mask] = 255
    
    return image

def create_checkerboard(size=(256, 256)):
    """Создание изображения с шахматным паттерном"""
    image = np.zeros(size, dtype=np.uint8)
    
    # Создаем шахматный паттерн
    square_size = 32
    for i in range(0, size[0], square_size):
        for j in range(0, size[1], square_size):
            if (i//square_size + j//square_size) % 2 == 0:
                image[i:i+square_size, j:j+square_size] = 255
    
    return image

def create_gradient(size=(256, 256)):
    """Создание изображения с градиентом и диагональными линиями"""
    # Создаем градиент
    x = np.linspace(0, 255, size[1])
    y = np.linspace(0, 255, size[0])
    X, Y = np.meshgrid(x, y)
    image = (X + Y) / 2
    
    # Добавляем диагональные линии
    thickness = 2
    spacing = 20
    for i in range(-size[0], size[0], spacing):
        indices = np.array([(x, x + i) for x in range(max(-i, 0), min(size[0], size[0]-i))])
        if len(indices) > 0:
            for t in range(thickness):
                valid_indices = (indices[:, 0] + t < size[0]) & (indices[:, 1] + t < size[1])
                image[indices[valid_indices, 0] + t, indices[valid_indices, 1] + t] = 255
    
    return image.astype(np.uint8)

def create_test_images():
    """Создание трех разных тестовых изображений"""
    # Создаем директорию для тестовых изображений, если её нет
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
    
    # Создаем и сохраняем первое изображение (полосы с кругом)
    img1 = create_striped_image()
    Image.fromarray(img1).save('test_images/test_image1.png')
    
    # Создаем и сохраняем второе изображение (шахматная доска)
    img2 = create_checkerboard()
    Image.fromarray(img2).save('test_images/test_image2.png')
    
    # Создаем и сохраняем третье изображение (градиент с линиями)
    img3 = create_gradient()
    Image.fromarray(img3).save('test_images/test_image3.png')
    
    print("Тестовые изображения созданы в директории 'test_images'")

if __name__ == "__main__":
    create_test_images()