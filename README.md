# Автоматическое построение признаков для ранжирования рекомендаций (Yambda)

> Черновик: отчёт будет дополнен результатами экспериментов.

Быстрый старт:
```bash
pip install -r requirements.txt
make test     # тесты на утечки
make smoke    # прогон на синтетике (~30 с)
make run      # основной эксперимент на Yambda-50M (скачивается с HuggingFace)
make scale report
```
На Kaggle: `notebooks/kaggle_run.ipynb` (Internet = On).
