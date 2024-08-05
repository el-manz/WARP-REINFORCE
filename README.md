# WARP-REINFORCE

## Как запускать решение?
В репозитории лежат файлы с необходимыми классами:
* `PrepareData` из файла [preparedata.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/preparedata.py): внутри конструктора создаёт обучающий датасет для модели наград, сохраняет его в `self.train_pairwise`
* `PreparePrompts` из файла [preparedata.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/preparedata.py): так же внутри конструктора создаёт обучающий датасет для генерирующей модели (которую будем обучать предложенным в статье методом), сохраняет его в `self.train_dataset` (валидационный сохраняет в `self.val_dataset`, но мы им не пользуемся).
* `RewardModel` из файла [rewardmodel.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/rewardmodel.py): задаёт модель наград с указанными параметрами. Параметры, которые обязательно нужно передать в конструктор - `train_dataset`, `val_dataset` и `num_train_epochs`. Для обучения необходимо вызвать метод `self.fit()`.
* `RewardDataCollatorWithPadding` и `RewardTrainer` из файла [rewardmodel.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/rewardmodel.py) - вспомогательные классы, задающие Data Collator и Trainer, использующиеся в дальнейшем в обучении модели наград.
* `WARP` из файла [warpmethod.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/warpmethod.py): задаёт метод WARP с выбранными параметрами. Для запуска цикла обучения необходимо вызвать метод `self.run_method()`, возвращающий новую обученную модель.
  
Вся последовательность решения, аналитика и выводы вынесены в ноутбук `Solution.ipynb`. Для полного воспроизведения решения достаточно запустить весь ноутбук: там уже прописаны импорты классов из этого репозитория. \
Модели и обучающие датасеты так же загруженны на huggingface (прикрепить ссылки), при необходимости можно загрузить их оттуда.
