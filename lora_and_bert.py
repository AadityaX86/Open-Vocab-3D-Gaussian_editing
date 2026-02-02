# ============================================================
# BERT + LoRA Token Classification (VS Code / Desktop Version)
# ============================================================

import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. Labels
# ------------------------------------------------------------
LABEL_LIST = [
    "O",
    "B-TARGET", "I-TARGET",
    "B-ACTION", "I-ACTION",
    "B-DIRECTION", "I-DIRECTION",
    "B-ORIENTATION", "I-ORIENTATION",
    "B-ATTRIBUTE", "I-ATTRIBUTE"
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

NUM_LABELS = len(LABEL_LIST)

# ------------------------------------------------------------
# 2. Dataset generator
# ------------------------------------------------------------
base_commands = [
    (
        ["Move","the","red","apple","to","the","right"],
        ["B-ACTION","O","B-ATTRIBUTE","B-TARGET","O","O","B-DIRECTION"]
    ),
    (
        ["Rotate","the","metal","box","counterclockwise","to","the","left"],
        ["B-ACTION","O","B-ATTRIBUTE","B-TARGET","B-ORIENTATION","O","O","B-DIRECTION"]
    ),
]

def expand_dataset(n=3000):
    actions = ["Move","Slide","Lift","Rotate","Push","Pull","Drop","Place"]
    objects = ["ball","box","chair","phone","book","bottle","bag"]
    colors = ["red","blue","green","black","white"]
    sizes = ["big","small"]
    directions = ["left","right","up","down"]
    orientations = ["clockwise","counterclockwise"]

    data = []

    for _ in range(n):
        tokens, labels = [], []

        act = random.choice(actions)
        tokens.append(act)
        labels.append("B-ACTION")

        tokens.append("the")
        labels.append("O")

        if random.random() < 0.6:
            tokens.append(random.choice(colors))
            labels.append("B-ATTRIBUTE")

        if random.random() < 0.3:
            tokens.append(random.choice(sizes))
            labels.append("I-ATTRIBUTE")

        obj = random.choice(objects)
        tokens.append(obj)
        labels.append("B-TARGET")

        if random.random() < 0.7:
            tokens += ["to", "the"]
            labels += ["O", "O"]
            tokens.append(random.choice(directions))
            labels.append("B-DIRECTION")

        if random.random() < 0.3:
            tokens.append(random.choice(orientations))
            labels.append("B-ORIENTATION")

        data.append((tokens, labels))

    return data

# build dataset
data = base_commands + expand_dataset(3000)

tokens = [x[0] for x in data]
labels = [[LABEL2ID[l] for l in x[1]] for x in data]

train_toks, val_toks, train_labs, val_labs = train_test_split(
    tokens, labels, test_size=0.1
)

train_ds = Dataset.from_dict({"tokens": train_toks, "labels": train_labs})
val_ds   = Dataset.from_dict({"tokens": val_toks,   "labels": val_labs})

# ------------------------------------------------------------
# 3. Tokenizer & model
# ------------------------------------------------------------
BASE_MODEL = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# ------------------------------------------------------------
# 4. Tokenization + label alignment
# ------------------------------------------------------------
def tokenize_and_align(example):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True
    )

    labels = []
    prev_word_id = None

    for word_id in tokenized.word_ids():
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(example["labels"][word_id])
        else:
            labels.append(-100)
        prev_word_id = word_id

    tokenized["labels"] = labels
    return tokenized

train_ds = train_ds.map(tokenize_and_align)
val_ds   = val_ds.map(tokenize_and_align)

# ------------------------------------------------------------
# 5. LoRA setup
# ------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "key", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type="TOKEN_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ------------------------------------------------------------
# 6. Training
# ------------------------------------------------------------
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir="./token_classifier_lora_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./token_classifier_lora_model")

# ------------------------------------------------------------
# 7. Reload model for inference (clean way)
# ------------------------------------------------------------
base_model = AutoModelForTokenClassification.from_pretrained(
    BASE_MODEL,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

model = PeftModel.from_pretrained(
    base_model,
    "./token_classifier_lora_model"
)

model.eval()

# ------------------------------------------------------------
# 8. Inference pipeline
# ------------------------------------------------------------
nlp = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# ------------------------------------------------------------
# 9. Test
# ------------------------------------------------------------
tests = [
    "Rotate the red box counterclockwise to the left",
    "Move the small bottle to the right",
    "Lift the blue chair"
]

for t in tests:
    print("\nINPUT:", t)
    print(nlp(t))
