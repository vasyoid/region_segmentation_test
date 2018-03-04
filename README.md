# Region Segmentation
Исследовательская работа сегментации фона на изображении.

## Описание
Утилита запускается из командной строки и принимает на вход имя файла, который необходимо сегментировать, имя файла, в котором содержится маска и имя выходного файла.
Маска должна быть такого же размера как и изоборажение.
Синим цветом должны быть выделены области, принадлежащие фону.
Красным цветом должны быть выделены области, принадлежащие объекту.
Остальные области маски должны быть белыми.

## Сборка
```bash
cmake .
cmake --build .
```
либо
```bash
make
```

## Использование
```bash
./run имя_файла_изображения имя_файла_маски имя_выходного_файла
```
## Результаты
На данный момент реализован модифицированный алгоритм, использующий сток и исток только для пикселей из маски.
Рассмотрены две версии -- с линейной и экспоненциальной зависимостью весов ребер от разницы цветов в пикселях.
Примеры можно найти в директории res.