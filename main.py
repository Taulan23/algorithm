from spiht_compression import SPIHT
import os
import sys
from PIL import Image
import numpy as np
from pathlib import Path

def validate_image(image_path):
    """Проверка валидности изображения"""
    try:
        with Image.open(image_path) as img:
            # Проверяем, что изображение открывается
            img.verify()
            
            # Повторно открываем изображение после verify()
            img = Image.open(image_path)
            
            # Проверяем минимальный размер
            if img.size[0] < 32 or img.size[1] < 32:
                raise ValueError("Изображение слишком маленькое (минимум 32x32 пикселей)")
            
            # Проверяем максимальный размер
            if img.size[0] > 4096 or img.size[1] > 4096:
                raise ValueError("Изображение слишком большое (максимум 4096x4096 пикселей)")
            
            # Проверяем, что изображение не повреждено
            img.load()
            
            return True
    except Exception as e:
        print(f"Ошибка при проверке изображения {image_path}: {str(e)}")
        return False

def process_images(input_paths, compression_ratio=8):
    """Обработка нескольких изображений"""
    results = []
    
    if not input_paths:
        print("Не указаны пути к изображениям!")
        return results
        
    # Создаем директорию для сжатых изображений, если её нет
    output_dir = "compressed_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for input_path in input_paths:
        try:
            print(f"\nОбработка изображения: {input_path}")
            
            # Проверка существования файла
            if not os.path.exists(input_path):
                print(f"Файл {input_path} не найден, пропускаем...")
                continue
            
            # Проверка, что это файл изображения
            if not input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                print(f"Файл {input_path} не является изображением, пропускаем...")
                continue
                
            if not validate_image(input_path):
                continue
                
            # Создаем экземпляр SPIHT
            spiht = SPIHT()
            
            try:
                # Сжатие изображения
                compressed, output_path = spiht.compress(input_path, compression_ratio)
                
                # Проверка результата
                if not os.path.exists(output_path):
                    raise RuntimeError("Ошибка при сохранении сжатого изображения")
                
                # Проверка размера сжатого файла
                if os.path.getsize(output_path) == 0:
                    raise RuntimeError("Размер сжатого файла равен 0")
                    
                # Сбор статистики
                original_size = os.path.getsize(input_path)
                compressed_size = len(compressed) // 8
                
                results.append({
                    'input_path': input_path,
                    'output_path': output_path,
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': original_size/compressed_size
                })
                
                # Визуализация
                try:
                    spiht.visualize_results(input_path, output_path)
                except Exception as viz_error:
                    print(f"Предупреждение: Не удалось создать визуализацию для {input_path}: {str(viz_error)}")
                
                print(f"Успешно обработано: {input_path}")
                
            except Exception as compress_error:
                print(f"Ошибка при сжатии {input_path}: {str(compress_error)}")
                continue
            
        except Exception as e:
            print(f"Ошибка при обработке {input_path}: {str(e)}", file=sys.stderr)
            continue
    
    if not results:
        print("\nВнимание: Ни одно изображение не было успешно обработано!")
    
    return results

def print_summary(results):
    """Вывод общей статистики по всем обработанным изображениям"""
    if not results:
        print("\nНет успешно обработанных изображений")
        return
        
    print("\n=== Итоговая статистика ===")
    print(f"Всего обработано изображений: {len(results)}")
    
    total_original = sum(r['original_size'] for r in results)
    total_compressed = sum(r['compressed_size'] for r in results)
    avg_ratio = sum(r['compression_ratio'] for r in results) / len(results)
    
    print(f"Общий исходный размер: {total_original:,} байт")
    print(f"Общий размер после сжатия: {total_compressed:,} байт")
    print(f"Средний коэффициент сжатия: {avg_ratio:.2f}")
    
    print("\nДетали по каждому изображению:")
    for r in results:
        print(f"\nФайл: {Path(r['input_path']).name}")
        print(f"  Исходный размер: {r['original_size']:,} байт")
        print(f"  Сжатый размер: {r['compressed_size']:,} байт")
        print(f"  Коэффициент сжатия: {r['compression_ratio']:.2f}")
        print(f"  Сохранено как: {Path(r['output_path']).name}")

def main():
    try:
        # Создаем тестовые изображения
        from create_test_image import create_test_images
        create_test_images()
        
        # Список изображений для обработки
        input_images = [
            "test_images/test_image1.png",
            "test_images/test_image2.png",
            "test_images/test_image3.png"
        ]
        
        # Проверяем существование всех файлов
        missing_files = [f for f in input_images if not os.path.exists(f)]
        if missing_files:
            print("Предупреждение: Следующие файлы не найдены:")
            for f in missing_files:
                print(f"  - {f}")
        
        # Задаем коэффициент сжатия
        compression_ratio = 8
        
        # Обработка изображений
        results = process_images(input_images, compression_ratio)
        
        # Вывод статистики
        print_summary(results)
            
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 