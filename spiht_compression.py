import numpy as np
from PIL import Image
import pywt
import matplotlib.pyplot as plt
import os

class SPIHT:
    def __init__(self):
        self.image = None
        self.coefficients = None
        self.threshold = None
        self.max_levels = 3
        
    def load_image(self, path):
        """Загрузка изображения и преобразование в градации серого"""
        try:
            self.image = Image.open(path).convert('L')
            # Убедимся, что размеры изображения являются степенью 2
            width, height = self.image.size
            new_width = 2**int(np.log2(width))
            new_height = 2**int(np.log2(height))
            if (width != new_width) or (height != new_height):
                self.image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return np.array(self.image)
        except Exception as e:
            raise ValueError(f"Ошибка при загрузке изображения: {str(e)}")
    
    def apply_dwt(self, data, levels=4):
        """Применение дискретного вейвлет-преобразования"""
        try:
            # Проверяем, что размеры изображения подходят для декомпозиции
            if data.size == 0:
                raise ValueError("Пустые данные изображения")
                
            min_size = min(data.shape)
            max_levels = pywt.dwt_max_level(min_size, 'bior2.2')
            levels = min(levels, max_levels)
            
            if levels < 1:
                raise ValueError("Невозможно выполнить декомпозицию с указанным количеством уровней")
            
            # Применяем вейвлет-преобразование
            coeffs = pywt.wavedec2(data, 'bior2.2', level=levels)
            
            # Сохраняем коэффициенты
            self.coefficients = []
            self.coefficients.append(coeffs[0])  # Аппроксимация
            
            # Сохраняем детализирующие коэффициенты
            for detail_coeffs in coeffs[1:]:
                self.coefficients.append(detail_coeffs)
            
            return self.coefficients
        except Exception as e:
            raise ValueError(f"Ошибка при применении DWT: {str(e)}")
    
    def _get_descendants(self, x, y, level):
        """Получение потомков для координат (x,y) на определенном уровне"""
        descendants = []
        if level < len(self.coefficients) - 1:
            # Вычисляем координаты потомков
            x_child = 2 * x
            y_child = 2 * y
            
            # Проверяем размеры следующего уровня
            next_level = level + 1
            if next_level < len(self.coefficients):
                shape = self.coefficients[next_level][0].shape if isinstance(self.coefficients[next_level], tuple) else self.coefficients[next_level].shape
                
                # Добавляем потомков только если они в пределах размеров
                if x_child + 1 < shape[0] and y_child + 1 < shape[1]:
                    descendants = [
                        (x_child, y_child, next_level),
                        (x_child, y_child+1, next_level),
                        (x_child+1, y_child, next_level),
                        (x_child+1, y_child+1, next_level)
                    ]
        return descendants
    
    def _is_significant(self, x, y, level):
        """Проверка значимости коэффициента"""
        try:
            if level == 0:
                return abs(self.coefficients[0][x, y]) >= self.threshold
            
            if isinstance(self.coefficients[level], tuple):
                coeff_h, coeff_v, coeff_d = self.coefficients[level]
                shape = coeff_h.shape
                
                if x >= shape[0] or y >= shape[1]:
                    return False
                
                # Проверяем все три типа коэффициентов
                coeffs = [
                    coeff_h[x, y] if y < shape[1] else 0,
                    coeff_v[x, y - shape[1]] if shape[1] <= y < 2*shape[1] else 0,
                    coeff_d[x, y - 2*shape[1]] if 2*shape[1] <= y < 3*shape[1] else 0
                ]
                return any(abs(c) >= self.threshold for c in coeffs)
            
            return abs(self.coefficients[level][x, y]) >= self.threshold
        except:
            return False
    
    def _sorting_pass(self, LIP, LSP, LIS, encoded_bits):
        """Улучшенный сортировочный проход"""
        # 1. Обработка LIP (List of Insignificant Pixels)
        i = 0
        while i < len(LIP):
            x, y, level = LIP[i]
            significant = self._is_significant(x, y, level)
            encoded_bits.append(1 if significant else 0)
            
            if significant:
                LSP.append(LIP.pop(i))
                # Получаем коэффициент с учетом уровня
                if level == 0:
                    coeff = self.coefficients[0][x, y]
                else:
                    if isinstance(self.coefficients[level], tuple):
                        coeff_h, coeff_v, coeff_d = self.coefficients[level]
                        h_shape = coeff_h.shape
                        
                        if x < h_shape[0]:
                            if y < h_shape[1]:
                                coeff = coeff_h[x, y]
                            else:
                                coeff = coeff_v[x, y - h_shape[1]]
                        else:
                            coeff = coeff_d[x - h_shape[0], y]
                    else:
                        coeff = self.coefficients[level][x, y]
                
                encoded_bits.append(1 if coeff > 0 else 0)
            else:
                i += 1

        # 2. Обработка LIS (List of Insignificant Sets)
        i = 0
        while i < len(LIS):
            if len(LIS[i]) == 4:
                x, y, level, type_A = LIS[i]
            else:
                x, y, level = LIS[i]
                type_A = True
            
            if type_A:
                descendants = self._get_descendants(x, y, level)
                significant = any(self._is_significant(dx, dy, dl) for dx, dy, dl in descendants)
                encoded_bits.append(1 if significant else 0)
                
                if significant:
                    for dx, dy, dl in descendants:
                        sig = self._is_significant(dx, dy, dl)
                        encoded_bits.append(1 if sig else 0)
                        
                        if sig:
                            LSP.append((dx, dy, dl))
                            # Получаем коэффициент с учетом уровня
                            if dl == 0:
                                coeff = self.coefficients[0][dx, dy]
                            else:
                                if isinstance(self.coefficients[dl], tuple):
                                    coeff_h, coeff_v, coeff_d = self.coefficients[dl]
                                    h_shape = coeff_h.shape
                                    
                                    if dx < h_shape[0]:
                                        if dy < h_shape[1]:
                                            coeff = coeff_h[dx, dy]
                                        else:
                                            coeff = coeff_v[dx, dy - h_shape[1]]
                                    else:
                                        coeff = coeff_d[dx - h_shape[0], dy]
                                else:
                                    coeff = self.coefficients[dl][dx, dy]
                                
                                encoded_bits.append(1 if coeff > 0 else 0)
                        else:
                            LIP.append((dx, dy, dl))
                    
                    if len(descendants) > 0:
                        LIS[i] = (x, y, level, False)
                    else:
                        LIS.pop(i)
                        i -= 1
            else:
                descendants = self._get_all_descendants(x, y, level)
                significant = any(self._is_significant(dx, dy, dl) for dx, dy, dl in descendants)
                encoded_bits.append(1 if significant else 0)
                
                if significant:
                    for dx, dy, dl in self._get_descendants(x, y, level):
                        LIS.append((dx, dy, dl, True))
                    LIS.pop(i)
                    i -= 1
            i += 1

    def _get_all_descendants(self, x, y, level):
        """Получение всех потомков для множества типа B"""
        descendants = []
        to_process = [(x, y, level)]
        
        while to_process:
            curr_x, curr_y, curr_level = to_process.pop(0)
            direct_descendants = self._get_descendants(curr_x, curr_y, curr_level)
            descendants.extend(direct_descendants)
            to_process.extend(direct_descendants)
            
        return descendants

    def _refinement_pass(self, LSP, encoded_bits):
        """Уточняющий проход"""
        for x, y, level in LSP:
            try:
                if level == 0:
                    coeff = self.coefficients[0][x, y]
                else:
                    if isinstance(self.coefficients[level], tuple):
                        coeff_h, coeff_v, coeff_d = self.coefficients[level]
                        shape = coeff_h.shape
                        
                        if y < shape[1]:
                            coeff = coeff_h[x, y]
                        elif y < 2 * shape[1]:
                            coeff = coeff_v[x, y - shape[1]]
                        else:
                            coeff = coeff_d[x, y - 2 * shape[1]]
                    else:
                        coeff = self.coefficients[level][x, y]
                
                # Проверяем значимый бит
                significant_bit = (abs(coeff) & self.threshold) != 0
                encoded_bits.append(1 if significant_bit else 0)
            except Exception as e:
                encoded_bits.append(0)

    def encode(self, target_size):
        """Кодирование изображения"""
        self._initialize_sets()
        
        # Инициализация списков
        LIP = [(x, y, 0) for x, y in self.T['T0']]
        LSP = []
        LIS = [(x, y, 0, True) for x, y in self.T['T0'] if self._get_descendants(x, y, 0)]
        
        # Инициализация порога
        max_coeff = max(abs(self.coefficients[0].max()), abs(self.coefficients[0].min()))
        self.threshold = 2**int(np.log2(max_coeff))
        
        encoded_bits = []
        
        while len(encoded_bits) < target_size and self.threshold >= 1:
            # Сортировочный проход
            self._sorting_pass(LIP, LSP, LIS, encoded_bits)
            
            # точняющий проход
            if LSP:
                self._refinement_pass(LSP, encoded_bits)
            
            self.threshold //= 2
        
        return encoded_bits
    
    def decode(self, encoded_bits, image_size):
        """Декодирование изображения"""
        # Создаем структуру для коэффициентов
        decoded_coeffs = []
        
        # Копируем структуру оригинальных коэффициентов
        for level in range(len(self.coefficients)):
            if level == 0:
                decoded_coeffs.append(np.zeros_like(self.coefficients[0]))
            else:
                if isinstance(self.coefficients[level], tuple):
                    h_shape = self.coefficients[level][0].shape
                    decoded_coeffs.append((
                        np.zeros(h_shape),  # Horizontal
                        np.zeros(h_shape),  # Vertical
                        np.zeros(h_shape)   # Diagonal
                    ))
                else:
                    decoded_coeffs.append(np.zeros_like(self.coefficients[level]))
        
        # Инициализация порога как в методе encode
        max_coeff = max(abs(self.coefficients[0].max()), abs(self.coefficients[0].min()))
        self.threshold = 2**int(np.log2(max_coeff))
        
        # Декодирование
        bit_index = 0
        while bit_index < len(encoded_bits) and self.threshold >= 1:
            # Восстановление значимых коэффициентов
            if bit_index + 1 < len(encoded_bits):
                x, y = self._get_coordinates(bit_index)
                if encoded_bits[bit_index]:
                    sign = encoded_bits[bit_index + 1]
                    value = self.threshold * (1 if sign else -1)
                    
                    # Определяем уровень и позицию
                    level = 0
                    if x < decoded_coeffs[0].shape[0] and y < decoded_coeffs[0].shape[1]:
                        decoded_coeffs[0][x, y] = value
                    else:
                        for l in range(1, len(decoded_coeffs)):
                            if isinstance(decoded_coeffs[l], tuple):
                                h_shape = decoded_coeffs[l][0].shape
                                if x < h_shape[0] and y < h_shape[1]:
                                    if y < h_shape[1]:
                                        decoded_coeffs[l][0][x, y] = value  # Horizontal
                                    elif y < 2*h_shape[1]:
                                        decoded_coeffs[l][1][x, y-h_shape[1]] = value  # Vertical
                                    else:
                                        decoded_coeffs[l][2][x, y-2*h_shape[1]] = value  # Diagonal
                                    break
                
                    bit_index += 2
                else:
                    bit_index += 1
            else:
                bit_index += 1
            
            # Уменьшаем порог
            if bit_index % 1000 == 0:
                self.threshold //= 2
        
        # Обратное вейвлет-преобразование
        try:
            reconstructed = pywt.waverec2(decoded_coeffs, 'bior2.2')
            # Обрезаем до исходного размера и нормализуем значения
            reconstructed = reconstructed[:image_size[0], :image_size[1]]
            reconstructed = np.clip(reconstructed, 0, 255)
            return reconstructed.astype(np.uint8)
        except Exception as e:
            print(f"Ошибка при обратном вейвлет-преобразовании: {str(e)}")
            return np.zeros(image_size, dtype=np.uint8)
    
    def compress(self, input_path, compression_ratio=8):
        """Полный про��есс сжатия изображения"""
        try:
            # Проверка входных параметров
            if compression_ratio < 1:
                raise ValueError("Коэффициент сжатия должен быть больше или равен 1")
                
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Файл {input_path} не найден")
                
            # Загрузка изображения
            img_data = self.load_image(input_path)
            
            # Применение DWT
            coeffs = self.apply_dwt(img_data, levels=4)
            
            # Определение целевого размера
            target_size = max((img_data.size * 8) // compression_ratio, 1024)  # Минимальный размер 1KB
            
            # Кодирование
            encoded = self.encode(target_size)
            
            # Сохранение результатов
            output_path = input_path.rsplit('.', 1)[0] + '_compressed.png'
            
            # Декодирование
            decoded_image = self.decode(encoded, img_data.shape)
            
            # Нормализация и сохранение
            decoded_image = np.clip(decoded_image, 0, 255)
            Image.fromarray(decoded_image.astype(np.uint8)).save(output_path)
            
            return encoded, output_path
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при сжатии изображения: {str(e)}")
    
    def visualize_results(self, original_path, compressed_path):
        """Визуализация результатов сжатия"""
        original = Image.open(original_path)
        compressed = Image.open(compressed_path)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        plt.imshow(original, cmap='gray')
        plt.title('Оригинальное изображение')
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(compressed, cmap='gray')
        plt.title('Сжатое изображение')
        plt.axis('off')
        
        plt.show()

    def _initialize_sets(self):
        """Инициализация множеств согласно иерархической структуре"""
        # Получаем размеры коэффициентов аппроксимации
        rows, cols = self.coefficients[0].shape
        
        # Инициализация словаря для хранения множеств T_i
        self.T = {}
        
        # T_0: все коэффициенты аппроксимации
        self.T['T0'] = [(i, j) for i in range(rows) for j in range(cols)]

    def _get_coordinates(self, bit_index):
        """Поучение координат для декодирования"""
        rows, cols = self.coefficients[0].shape
        x = (bit_index // cols) % rows
        y = bit_index % cols
        return x, y

def main():
    # Пример использования
    spiht = SPIHT()
    
    # Путь к тестовому изображению
    input_image = "test_image.png"
    
    # Сжатие изображения
    compressed, output_path = spiht.compress(input_image, compression_ratio=8)
    
    # Вывод результатов
    original_size = os.path.getsize(input_image)
    compressed_size = len(compressed) // 8
    print(f"Исходный размер: {original_size} байт")
    print(f"Сжатый размер: {compressed_size} байт")
    print(f"Коэффициент сжатия: {original_size/compressed_size:.2f}")
    
    # Визуализация результатов
    spiht.visualize_results(input_image, output_path)

if __name__ == "__main__":
    main() 