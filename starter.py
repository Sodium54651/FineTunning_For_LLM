from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import torch, os, threading
print("Загружаюсь..")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
try:
    curdir = os.path.dirname(os.path.abspath(__file__))
    model_name = curdir + r"\FluffleMothLearned"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Квантование для ускорения на процессоре
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="cpu")
    # model = AutoModelForCausalLM.from_pretrained("gpt2")  # или другая базовая арх-ра

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
except Exception as ex:
    print("🛑Ошибочка", ex)
    exit



dialogue = ""
print("✅ Модель загружена! Можно общаться.")
while True:
    # записываем данные
    user_input = input("\nТы: ")

    with open(f"{curdir}\Logs.txt", 'a', encoding='utf-8') as fin:
        dialogue += f"User: {user_input}\n"
        fin.write(f"?User: {user_input}\n\n")

    if user_input.lower() in ["выход", "exit", "quit"]:
        break

    

    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, max_length=512)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,                     # токены и attention mask
            streamer=streamer,            # стриминг ответа
            max_new_tokens=128,           # макс. длина ответа, больше → длиннее ответ
            min_length=1,                # мин. длина ответа, больше → ответ длиннее
            do_sample=True,               # случайная генерация, True → больше креатива
            top_k=50,                      # выбор токенов из топ‑K, больше → больше вариативности
            top_p=0.9,                     # nucleus sampling, больше → более случайный ответ
            temperature=0.8,               # творческость, больше → креативнее
            repetition_penalty=1.2,       # штраф за повтор, больше → меньше повторений
            no_repeat_ngram_size=2,       # запрещает повторение 2‑словных фраз, больше → меньше повторов
            length_penalty=1.0,           # штраф за длину ответа, больше → длиннее ответы
            num_return_sequences=1        # сколько вариантов ответа вернуть
    )
    )

    thread.start()

    print("\n🤖:", end="", flush=True)
    full_reply = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        full_reply += new_text
    thread.join()

    dialogue += f"AI: {full_reply}\n"
    with open(f"{curdir}\Logs.txt", 'a', encoding='utf-8') as fin:
        fin.write(f"!AI: {full_reply}\n\n")
    print()
    # print("\nдиологи" + dialogue)
