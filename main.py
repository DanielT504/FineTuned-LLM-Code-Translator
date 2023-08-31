from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

class CustomDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

def log_and_check_data(data, name):
    print(f"{name} size: {len(data)}")
    print(f"Sample {name} data: {data[:3]}")

train_python = read_file("XLCoST_data/generation/pair_data_tok_1/Python-Javascript/train-Python-Javascript-tok.py")
train_javascript = read_file("XLCoST_data/generation/pair_data_tok_1/Python-Javascript/train-Python-Javascript-tok.js")
log_and_check_data(train_python, "Train Python")
log_and_check_data(train_javascript, "Train JavaScript")

eval_python = read_file("XLCoST_data/generation/pair_data_tok_1/Python-Javascript/test-Python-Javascript-tok.py")
eval_javascript = read_file("XLCoST_data/generation/pair_data_tok_1/Python-Javascript/test-Python-Javascript-tok.js")
log_and_check_data(eval_python, "Eval Python")
log_and_check_data(eval_javascript, "Eval JavaScript")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
quick_test_size = 1000
train_encodings = tokenizer(train_python[:quick_test_size] + train_javascript[:quick_test_size], truncation=True, padding=True)
eval_encodings = tokenizer(eval_python[:quick_test_size] + eval_javascript[:quick_test_size], truncation=True, padding=True)
# train_encodings = tokenizer(train_python + train_javascript, truncation=True, padding=True)
# eval_encodings = tokenizer(eval_python + eval_javascript, truncation=True, padding=True)
print(f"Train encodings keys: {train_encodings.keys()}")
print(f"Eval encodings keys: {eval_encodings.keys()}")

train_dataset = CustomDataset(train_encodings)
eval_dataset = CustomDataset(eval_encodings)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

model = AutoModelForCausalLM.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir='./results',
    save_strategy="steps",
    save_steps=200,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_first_step=True,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
trainer.evaluate()

prompt = "translate this Python code to JavaScript: print('Hello, world!')"
input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']
attention_mask = tokenizer(prompt, return_tensors='pt')['attention_mask']

output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=50, pad_token_id=tokenizer.eos_token_id)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated text: ", output_text)

model.save_pretrained('./my_finetuned_model')
