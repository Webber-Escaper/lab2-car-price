# Данные

Файлы данных намеренно не хранятся в Git-репозитории.

Исходный датасет: **Car Price Prediction — Used Cars**  
Источник: https://www.kaggle.com/datasets/vijayaadithyanvg/car-price-predictionused-cars/data

После скачивания исходный CSV-файл необходимо сохранить в этой директории под именем:

```text
data/car_data.csv
```

Исходный файл в архиве Kaggle обычно называется `car data.csv`; его следует переименовать в `car_data.csv`.

После выполнения `eda/eda.ipynb` очищенная выборка будет сохранена как:

```text
data/clean_data.pkl
```

Файлы `*.csv` и `*.pkl` исключены из Git с помощью `.gitignore`.
