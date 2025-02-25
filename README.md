# WARP-REINFORCE

Реализация метода WARP из статьи [WARP: On the Benefits of Weight Averaged Rewarded Policies](https://arxiv.org/pdf/2406.16768) для задачи Reinforcement Learning from Human Feedback.\
Задача: сгенерировать положительные отзывы на фильмы на основе датасета [imdb](https://huggingface.co/datasets/stanfordnlp/imdb).

## Как запускать решение полностью?
В репозитории лежат файлы с необходимыми классами:
* `PrepareData` из файла [preparedata.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/preparedata.py): внутри конструктора создаёт обучающий датасет для модели наград, сохраняет его в `self.train_pairwise`
* `PreparePrompts` из файла [preparedata.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/preparedata.py): так же внутри конструктора создаёт обучающий датасет для генерирующей модели (которую будем обучать предложенным в статье методом), сохраняет его в `self.train_dataset` (валидационный сохраняет в `self.val_dataset`, но мы им не пользуемся).
* `RewardModel` из файла [rewardmodel.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/rewardmodel.py): задаёт модель наград с указанными параметрами. Параметры, которые обязательно нужно передать в конструктор - `train_dataset`, `val_dataset` и `num_train_epochs`. Для обучения необходимо вызвать метод `self.fit()`.
* `RewardDataCollatorWithPadding` и `RewardTrainer` из файла [rewardmodel.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/rewardmodel.py) - вспомогательные классы, задающие Data Collator и Trainer, использующиеся в дальнейшем в обучении модели наград.
* `WARP` из файла [warpmethod.py](https://github.com/el-manz/WARP-REINFORCE/blob/main/warpmethod.py): задаёт метод WARP с выбранными параметрами. Для запуска цикла обучения необходимо вызвать метод `self.run_method()`, возвращающий новую обученную модель.
  
Вся последовательность решения, аналитика и выводы вынесены в ноутбук [Solution.ipynb](https://github.com/el-manz/WARP-REINFORCE/blob/main/Solution.ipynb). Для полного воспроизведения решения достаточно запустить весь ноутбук: там уже прописаны импорты классов из этого репозитория.

## Как воспользоваться уже обученными моделями/собранными датасетами?
Модели и обучающие датасеты также загруженны на huggingface:
* [репозиторий с моделью наград](https://huggingface.co/elmanz/reward-model)
* [репозиторий с генерирующей моделью](https://huggingface.co/elmanz/rl-model)
* [обучающий датасет для модели наград](https://huggingface.co/datasets/elmanz/reward-model)
* [обучающий датасет для генерирующей модели](https://huggingface.co/datasets/elmanz/rl-model)

При необходимости можно загрузить их оттуда с помощью функций:
* для датасетов:
```
from datasets import load_dataset

ds = load_dataset("elmanz/dataset_name")
```
* для модели наград
```
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("elmanz/reward-model")
model = AutoModelForSequenceClassification.from_pretrained("elmanz/reward-model")
```
или через pipeline:
```
from transformers import pipeline

pipe = pipeline("text-classification", model="elmanz/reward-model")
```
* для генерирующей модели:
```
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("elmanz/rl-model")
model = AutoModelForCausalLM.from_pretrained("elmanz/rl-model")
```
или через pipeline:
```
from transformers import pipeline

pipe = pipeline("text-generation", model="elmanz/rl-model")
```
