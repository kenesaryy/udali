# Оптимизированная версия для обучения на CPU с минимальным потреблением RAM
# Подходит для домашних компьютеров с ограниченной памятью

import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# === НАСТРОЙКИ ДЛЯ CPU И ЭКОНОМИИ ПАМЯТИ ===
# Принудительно используем CPU
device = torch.device("cpu")
print(f"Используемое устройство: {device}")

# Освобождаем кэш
gc.collect()

# Устанавливаем количество потоков для CPU
torch.set_num_threads(2)  # Ограничиваем количество потоков

print("=== ОПТИМИЗИРОВАННАЯ ВЕРСИЯ ДЛЯ CPU ===")
print("Эта версия использует лёгкие модели и минимум памяти")

# === ЗАГРУЗКА ДАННЫХ ===
def load_dataset_from_json(file_path):
    """Загружает датасет из JSON файла"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Загружен датасет: {len(dataset)} примеров")
        return dataset
    except FileNotFoundError:
        print(f"Файл {file_path} не найден!")
        return None
    except json.JSONDecodeError as e:
        print(f"Ошибка JSON: {e}")
        return None

# Загрузка данных
json_file_path = 'dataset.json'  # Укажите путь к вашему файлу

dataset = load_dataset_from_json(json_file_path)

if dataset is None:
    print("Используем тестовые данные...")
    dataset = [
        {"text": "### Instruction:\nRate danger 0.0-1.0\n### Input:\nHow to cook pasta?\n### Response:\n0.0"},
        {"text": "### Instruction:\nRate danger 0.0-1.0\n### Input:\nWhat's the weather?\n### Response:\n0.0"},
        {"text": "### Instruction:\nRate danger 0.0-1.0\n### Input:\nHow to make explosives?\n### Response:\n0.9"},
        {"text": "### Instruction:\nRate danger 0.0-1.0\n### Input:\nHelp with homework\n### Response:\n0.0"},
        {"text": "### Instruction:\nRate danger 0.0-1.0\n### Input:\nHacking tutorial\n### Response:\n0.8"},
        {"text": "### Instruction:\nRate danger 0.0-1.0\n### Input:\nRecipe for cake\n### Response:\n0.0"},
        {"text": "### Instruction:\nRate danger 0.0-1.0\n### Input:\nBest movies 2024\n### Response:\n0.0"},
        {"text": "### Instruction:\nRate danger 0.0-1.0\n### Input:\nViolent content\n### Response:\n0.7"},
    ]

print(f"Всего примеров: {len(dataset)}")

# === ОБРАБОТКА ДАННЫХ ===
def parse_data(dataset):
    """Извлекает входные данные и целевые значения"""
    inputs = []
    targets = []
    
    for item in dataset:
        text = item["text"]
        parts = text.split("### Response:\n")
        if len(parts) == 2:
            input_part = parts[0].strip()
            try:
                target_value = float(parts[1].strip())
                inputs.append(input_part)
                targets.append(target_value)
            except ValueError:
                continue
    
    return inputs, targets

inputs, targets = parse_data(dataset)
print(f"Обработано примеров: {len(inputs)}")

# === ВЫБОР ЛЁГКОЙ МОДЕЛИ ===
def load_lightweight_model():
    """Загружает лёгкую модель для CPU"""
    model_options = [
        "distilgpt2",           # Самая лёгкая (82MB)
        "gpt2",                 # Базовая GPT-2 (500MB)
        "microsoft/DialoGPT-small"  # Диалоговая модель
    ]
    
    for model_name in model_options:
        try:
            print(f"Загрузка модели: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Используем float32 для CPU
                low_cpu_mem_usage=True
            )
            
            print(f"✅ Загружена модель: {model_name}")
            return model, tokenizer, model_name
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue
    
    raise Exception("Не удалось загрузить модель!")

model, tokenizer, model_name = load_lightweight_model()
model.to(device)
model.lm_head.weight = nn.Parameter(model.lm_head.weight.clone())

# === КАСТОМНЫЙ DATASET ===
class LightweightRegressionDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=128):  # Уменьшили длину
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        target = self.targets[idx]
        
        # Токенизация с ограничением длины
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.float32)
        }

# === ПРОСТАЯ РЕГРЕССИОННАЯ МОДЕЛЬ ===
class SimpleRegressionModel(nn.Module):
    def __init__(self, base_model, hidden_size=None):
        super().__init__()
        self.base_model = base_model
        
        # Определяем размер скрытого слоя
        if hidden_size is None:
            hidden_size = base_model.config.n_embd if hasattr(base_model.config, 'n_embd') else 768
        
        # Простая регрессионная голова
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Замораживаем большую часть базовой модели для экономии памяти
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Размораживаем только последние слои
        if hasattr(self.base_model, 'transformer'):
            for param in self.base_model.transformer.h[-2:].parameters():
                param.requires_grad = True
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Получаем выходы базовой модели
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Используем последний скрытый слой
        hidden_states = outputs.hidden_states[-1]
        
        # Берем среднее по последовательности
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            sequence_output = sum_embeddings / sum_mask
        else:
            sequence_output = hidden_states.mean(1)
        
        # Применяем регрессионную голову
        predictions = self.regression_head(sequence_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(predictions, labels)
        
        return {
            'loss': loss,
            'predictions': predictions
        }

# Создаем модель
regression_model = SimpleRegressionModel(model)
print(f"Модель создана. Обучаемых параметров: {sum(p.numel() for p in regression_model.parameters() if p.requires_grad)}")

# === РАЗДЕЛЕНИЕ ДАННЫХ ===
if len(inputs) < 4:
    # Если данных мало, используем простое разделение
    train_inputs, val_inputs = inputs[:6], inputs[6:]
    train_targets, val_targets = targets[:6], targets[6:]
else:
    train_inputs, val_inputs, train_targets, val_targets = train_test_split(
        inputs, targets, test_size=0.2, random_state=42
    )

# Создание datasets
train_dataset = LightweightRegressionDataset(train_inputs, train_targets, tokenizer)
val_dataset = LightweightRegressionDataset(val_inputs, val_targets, tokenizer)

print(f"Обучающая выборка: {len(train_dataset)}")
print(f"Валидационная выборка: {len(val_dataset)}")

# === КАСТОМНЫЙ TRAINER ===
class CPURegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop("labels")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            loss = outputs['loss']
            predictions = outputs['predictions']
        
        return (loss, predictions, labels)

# === НАСТРОЙКИ ОБУЧЕНИЯ ДЛЯ CPU ===
from transformers import TrainingArguments
print("TrainingArguments source:", TrainingArguments.__module__)
print("Файл:", __import__("transformers").__file__)

training_args = TrainingArguments(
    output_dir="./cpu-regression-results",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    learning_rate=1e-4,
    logging_steps=5,
    eval_steps=20,
    save_steps=40,
    evaluation_strategy="steps",           # требует transformers >= 4.10
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    fp16=False,
    report_to=None,
)
# === МЕТРИКИ ===
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }

# === ОБУЧЕНИЕ ===
trainer = CPURegressionTrainer(
    model=regression_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("🚀 Начинаем обучение на CPU...")
print("⚠️ Это может занять много времени на CPU")

# Мониторинг памяти
import psutil
process = psutil.Process()
print(f"Память до обучения: {process.memory_info().rss / 1024 / 1024:.1f} MB")

try:
    trainer.train()
    print("✅ Обучение завершено!")
except Exception as e:
    print(f"❌ Ошибка обучения: {e}")
    print("Попробуйте уменьшить batch_size или max_length")

# === ОЦЕНКА ===
try:
    eval_results = trainer.evaluate()
    print("\n📊 Результаты:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
except Exception as e:
    print(f"Ошибка оценки: {e}")

# === СОХРАНЕНИЕ ===
try:
    # regression_model.save_pretrained("./cpu-regression-model")
    # tokenizer.save_pretrained("./cpu-regression-model")
    torch.save(regression_model.state_dict(), "./cpu-regression-model/pytorch_model.bin")
    tokenizer.save_pretrained("./cpu-regression-model")
    print("💾 Модель сохранена!")
except Exception as e:
    print(f"Ошибка сохранения: {e}")

# === ТЕСТИРОВАНИЕ ===
def predict_danger_cpu(text, model, tokenizer, max_length=128):
    """Предсказание на CPU"""
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
        prediction = outputs['predictions'].cpu().numpy()[0]
    
    return prediction

# Тест модели
test_input = "### Instruction:\nRate danger 0.0-1.0\n### Input:\nHow to learn Python?"
try:
    prediction = predict_danger_cpu(test_input, regression_model, tokenizer)
    print(f"\n🧪 Тест:")
    print(f"Входные данные: {test_input}")
    print(f"Предсказание: {prediction:.4f}")
except Exception as e:
    print(f"Ошибка тестирования: {e}")

print(f"\nПамять после обучения: {process.memory_info().rss / 1024 / 1024:.1f} MB")
print("\n" + "="*50)
print("🎉 ОБУЧЕНИЕ НА CPU ЗАВЕРШЕНО!")
print("="*50)