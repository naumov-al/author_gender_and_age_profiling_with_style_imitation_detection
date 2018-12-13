# Демоверсии компьютерной программы для диагностирования пола и возраста участника интернет-коммуникации на основе количественных параметров его текстов

Основные модули:
- train.py -- модуль для обучения модели;
- predict.py -- модуль предсказания класс для текста;
- text_to_json.py -- модуль морфологической обработки текста на основе модели UDPipe;
- eval.py -- модель для оценки точности модели на заданном корпусе.

Список необходимых сторонних библиотек приведён в файле requirements.txt.
Модель работают в ОС Ubuntu 16.04 64-bit.

Модели:
- `models/res_model_gender_nnmodel/` -- модель на основе нейронных сетей для определения пола автора текста, обучалась на текстах без сокрытия пола;
- `models/res_model_gender_charngram.pkl` -- модель на основе метода Gradient Boosting и кодирования текста как мешок н-грамм (от 3-8 грамм) символов, обучена на текстах без сокрытия пола;
- `models/res_model_gender_cs_no_im_gender_im.pkl` -- модель на основе метода Gradient Boosting и кодирования текста как мешок н-грамм символов (3-8 грамм), обучена на текстах без сокрытия пола и с имитацией противоположенного пола;
- `models/cs_age_imitation.pkl` -- модель для определения возрастной группы, на основе метода Gradient Boosting и кодирования текста как мекшо н-грамм символов (3-8 грамм), обучена на текстах без имитации возраста, с имитацией более молодого возраста, с имитацией более старшего возраста;
- `models/cs_age_imitation_imitation_type_model.pkl` -- модель для определния типа имитации возраста (no_im -- без имитации, younger -- имитация стиля младшего возраста, older -- имитация стиля старшего возраста) на основе мешка н-грамм символов (3-8 грамм) и метода классификации Gradient Boosting, обучение производилось на корпусе crowd source.

Модели cs_age_imitation.pkl и cs_age_imitation_imitation_type_model.pkl доступны по ссылке https://cloud.mail.ru/public/KW1N/H7z1NoMCF

В задаче определения пола класс 0 -- это женский пол, класс 1 -- это мужской пол автора текста.
В задаче определения возрастной группы: 20-30 это принадлежность автора к возрастной группе от 20 до 29 (включительно) лет, 30-40, 40-50 аналогично.

Для оценки произвольных текстов моделей необходимо сделать следующие шаги:
1. Подготовить csv файл (my_data.csv) с разделителем ;, первая строчка -- подписи к колонкам, в файле обязательно должна быть колонка text -- в ней должен быть текст докумуентов для анализа;
2. Запустить модуль csv_to_json.py командой: `python3 csv_to_json.py my_data.csv my_data.json ru_syntagrus.udpipe`, обученную модель UDPipe -- ru_syntagrus.udpipe можно взять из CoNLL 2018 Shared Task - UDPipe Baseline Models and Supplementary Materials, https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2859.
3. Запустить модуль предсказания класса для каждого текста: `python3 predict.py my_data.json models/res_model_gender_charngram.pkl --model-type CharNgram` в консоле будет отображён текст документа и предсказанный класс.  

Работа выполнена при поддержке гранта РНФ №16-18-10050 на тему: "Диагностирование пола и возраста участника интернет-коммуникации на основе количественных параметров его текстов".