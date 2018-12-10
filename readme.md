# Компьютерная программа для диагностирования пола и возраста автора текста с учетом возможного искажения признаков письменной речи с оценкой их эффективности

Основные модули:
- train.py -- модуль для обучения модели;
- predict.py -- модуль предсказания класс для текста;
- text_to_json.py -- модуль морфологической обработки текста на основе модели UDPipe;
- eval.py -- модель для оценки точности модели на заданном корпусе.