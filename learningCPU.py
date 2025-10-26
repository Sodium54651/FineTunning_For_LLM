import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


# отключаем всё, что не обучение
os.environ["WANDB_DISABLED"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"




model_path = os.path.dirname(os.path.abspath(__file__)) + r"\FluffleMoth"          #модель обучаемая
data_path  = os.path.dirname(os.path.abspath(__file__)) + r"\data.txt"             #данные для обучения
save_path  = os.path.dirname(os.path.abspath(__file__))  + r"\FluffleMothLearned"   #обученная модель

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
dataset   = load_dataset('text', data_files={'train': data_path})

def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

training_args = TrainingArguments(
    output_dir=save_path,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # уменьшить до 2 если OOM
    fp16=False,
    save_strategy="no",                   # никаких сохранений кроме финального
    logging_strategy="epoch",                # никаких логов
    dataloader_num_workers=0  # важно для Windows, чтобы не было крэша
)
print("[INFO] Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print("[INFO] Training complete ✅")
    input("⚡Модель обучилась нажмите любую клавишу...")



