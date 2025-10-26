import os
print("Импортирование библиотек...")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers import AutoModelForCausalLM as AMFS1 #1 тип библитек
from transformers import AutoModelForSeq2SeqLM as AMFS2  #2 тип библиотек
from datasets import load_dataset
import torch












    
# функция тестирования
def test_loss():
    # Инициализируем тренера для тестирования
    test_trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_test_datasets["test"]
    )
    
    # Запускаем оценку модели на тестовом наборе данных
    # metrics будет содержать различные метрики, включая loss
    metrics = test_trainer.evaluate()
    
    # Получаем значение loss из метрик
    test_loss = metrics['eval_loss']
    
    # Выводим текущее значение loss
    print(f"📊  Текущий показатель loss: {test_loss}")
    
    # Возвращаем значение loss для дальнейшего использования
    return test_loss





# 🧊блок выбора устройства

# выбор устройства cpu or cuda
svich_dev = "cuda" if torch.cuda.is_available() else "cpu"
# задаём параметры для оптимизатора вещ чисел только для cuda ядер
if svich_dev == "cuda":
    fp = True
else:
    fp = False











# 🧊блок загрузки данных

print("Загрузка началась...")
# прописание путей к данным и параметров в ручном режиме
epoch = 3
testYes = True
tokensSize = 256      # 512 ~ 3000 слов и знаков без пробелов

curdir = os.path.dirname(os.path.abspath(__file__))
model_path = curdir + r"\QA_Inator"          #модель обучаемая
data_path = curdir + r"\QA_Inator_data.txt"     #данные для обучения
test_path = curdir + r"\QA_Inator_test.txt"     #данные для тестирования не обязателен
save_path = curdir + r"\QA_Inator_learned"  #обученная модель









# 🧊блок токинезации

print("📝  Загрузка токенизатора...")
# токинизация мы берём токенизатор из модели попути что прописали выше, пусть там ищет
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
print("💾  Загрузка модели...")
# model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
try:
    model = AMFS1.from_pretrained(model_path, local_files_only=True)
    print("⚠️  используется 1 тип библиотек")
except Exception:
    print("⚠️  смена библиотеки на 2 тип:")
    model = AMFS2.from_pretrained(model_path, local_files_only=True)
model.to(svich_dev)
print("⚙️  Используем устройство: ", model.device)


#⬇️ подблок анализа данных

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    # инициализируем переменную для хранения максимального числа токенов в строке
    big_line = 0
    for line in lines:
        encoded = tokenizer(line, add_special_tokens=False)
        tokens = len(encoded["input_ids"])
        if big_line < tokens:
            big_line = tokens

    totalTokens = big_line
    tokensSize = totalTokens + 10

print(f"📊  Среднее по больнице количество токенов: {totalTokens}")
print(f"📏  Оптимальный размер токенов выбран: {tokensSize}")

# ⬇️под блок работы с датасетом и его токинезация

print("🗄️  Загрузка датасета...")
# загружаем данные что мы подготовили
dataset = load_dataset("text", data_files={"train": data_path})
try:
    testdataset = load_dataset("text", data_files={"test": test_path})
except Exception as e:
    print("⚠️  Ошибка при загрузке тестового датасета:", e)
    testYes = False

# функция что будет выполнять токенизацию нового текста
def tokenize(text):
    model_inputs = tokenizer(text["text"], truncation=True, padding="max_length", max_length=tokensSize)
    # labels для обучения (то, что модель должна предсказывать)
    # тут надо посмотреть какой формат она ждёт 
    try:
        # это строка что возвращает строку как вопрос и строку как ответ
        model_inputs["labels"] = tokenizer(text["text"], truncation=True, padding="max_length", max_length=tokensSize)["input_ids"]
    except Exception as e:
        # эта строка возвращает только 1 строчку и хватит
        model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

print("📝🗄️  Выполняется токенизация датасета...")
tokenized_datasets = dataset.map(tokenize, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# загрузка тестового датасета
if testYes:
    print("📝🗄️  Выполняется токенизация тестового датасета...")
    tokenized_test_datasets = testdataset.map(tokenize, batched=True)
    tokenized_test_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    












# 🧊подбор лучшей конфигурации

# fp отвечает за cuda ядра
if fp:
    print("🛢️  Выпоняется подбор оптимальных бачей")
    best_batch = None
    for b in [128, 64, 32, 16, 8, 4, 2, 1]:
        print(f"🔎  Поиск оптимального размера бача {b}")
        try:
        # выбор конфигурации
            training_args = TrainingArguments(
                output_dir=save_path,
                overwrite_output_dir=True,
                per_device_train_batch_size=b,
                fp16=fp,
                save_strategy="no",
                logging_strategy="no",
                dataloader_num_workers=0,
                max_steps=3
            )
        # делаем маленький тест — пару шагов

            trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            )

            trainer.train()
            best_batch = b
            print(f"✅  Подходит batch_size={b}")
            break

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"⚠️  batch_size={b} не влез, уменьшаем...")
                torch.cuda.empty_cache()
            else:
                raise e

    if best_batch is None:
        best_batch = 1  # на всякий случай
    print(f"🏁  Финальный batch_size={best_batch}")

    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=best_batch,
        fp16=fp,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=500,
        dataloader_num_workers=0,
    )


# если у тебя процессор и 16 гб RAM
else:
    #⚙️  задание параметров для обучения
    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        # num_train_epochs=3,
    # размер куся, это значает, что за 1 присест обучалка будет брать вот столько строчек
    # и на этих строчках дебать backward, и каждое увеличение сокращает время в 2 раза
    # но и память тоже нагружает в 2 раза, а так же чем больше батч тем лучше обучается нейросеть
    # batch больше = больше RAM и лучше усваение материала
        per_device_train_batch_size=2,
    # оптимизатор для обучения на видеокарте 
        fp16=fp,
    # методы логирования
        save_strategy="no",
        logging_strategy="steps",
    # говорит через сколько будет шаг
        logging_steps=10,
        dataloader_num_workers=0  # эта штука для винды должа быть она делает хорошо
    )
    print(f"🏁  Финальный batch_size={2}")

# инициализация тренера, указания модели, аргументы, и датасета
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)











# 🧊блок запуска

print("Начало обучения...")
try:
    for rh in range(epoch):
        trainer.train()
        if testYes:
            lert = test_loss()
            if  lert < 1.5:
                print(f"{epoch}, {current_loss}")
                print("🏆  Обучение завершено досрочно!")
                break
        else:
            if len(trainer.state.log_history) > 0:
                last_log = trainer.state.log_history[-1]
                if "loss" in last_log:
                    current_loss = last_log["loss"]
                    if current_loss < 1.0:
                        print(f"{epoch}, {current_loss}")
                        print("🏆  Обучение завершено досрочно!")
                        break

    print("обучение завершено")
    print("Сохраняем модель")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("🟩 модель сохранена")
    input("⚡ Модель обучилась нажмите любую клавишу...")
except KeyboardInterrupt:
    print("\n🟥 Обучение прервано вручную. Сохраняем модель...")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("🟩 модель сохранена")
    # input("✅ Модель сохранилась, нажмите любую клавишу...")