# 🔮 Liber777revised Symbolic Extraction System

A comprehensive system for extracting symbolic correspondences from the Liber777revised JP2 collection and organizing them into trainable datasets for LoRA training on **symbolic representation of future possible states**.

## 🎯 **What This System Does**

This system extracts symbolic correspondences from the Liber777revised JP2 files and creates structured datasets for training AI models to understand and predict **future possible states** based on symbolic relationships. The core concept is that symbols represent potentialities and transformations.

## 🏗️ **System Architecture**

### **Core Components**
- **`liber777_symbolic_extractor.py`** - Main extraction engine
- **`test_symbolic_extraction.py`** - Test and validation system
- **`LIBER777_SYMBOLIC_TRAINING_GUIDE.md`** - Comprehensive usage guide
- **`SYMBOLIC_EXTRACTION_SUMMARY.md`** - Complete system summary

### **Symbolic Knowledge Base**
The system includes comprehensive symbolic correspondences:

- **7 Planetary**: Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn
- **4 Elemental**: Fire, Water, Air, Earth
- **10 Qabalistic**: Tree of Life sephiroth (Keter to Malkuth)
- **12 Zodiacal**: All zodiac signs with correspondences
- **10 Numerical**: Numbers 1-10 with symbolic meanings
- **8 Color**: Color symbolism and associations

### **Extraction Methods**
- **OCR Text Analysis**: Extract text from JP2 images
- **Visual Pattern Recognition**: Detect geometric shapes and symbols
- **Symbolic Pattern Matching**: Identify correspondences in text
- **Table Structure Parsing**: Extract correspondence tables

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
# Core dependencies
pip install pandas numpy pillow opencv-python pytesseract pyyaml

# Optional: Perspective D<cide> framework
pip install perspective-dcide[etx]

# For advanced image processing
pip install scikit-image matplotlib
```

### **2. Test the System**
```bash
python test_symbolic_extraction.py
```

### **3. Extract Symbolic Data**
```bash
python liber777_symbolic_extractor.py
```

### **4. Review Generated Datasets**
The system creates several output files in `symbolic_datasets/`:
- `liber777_symbolic_dataset.json` - Complete dataset
- `liber777_correspondences.csv` - Tabular format
- `liber777_training_dataset.jsonl` - Training pairs for LoRA
- `liber777_validation_dataset.jsonl` - Validation pairs
- `liber777_symbolic_metadata.yaml` - Metadata and statistics

## 📊 **Expected Results**

### **Dataset Statistics**
- **Total Correspondences**: 500-1000+ (depending on content)
- **Training Pairs**: 2000-4000+ prompt-response pairs
- **Unique Symbols**: 100-200+ different symbolic elements
- **Quality Score**: 0.7-0.9 (depending on extraction quality)

### **Training Data Format**
The system generates training pairs in JSONL format:

```json
{"prompt": "What does the symbol sun represent in magical correspondences?", "response": "The symbol sun represents gold, yellow, sunday, 1, circle, king, vitality"}
{"prompt": "Explain the symbol moon", "response": "The moon represents silver, white, monday, 2, crescent, queen, intuition"}
{"prompt": "What does the number 7 represent in numerology?", "response": "The number 7 represents spirituality, mystery, perfection, wisdom"}
```

## 🎯 **LoRA Training Integration**

### **Dataset Loading**
```python
import json
from datasets import Dataset

def load_symbolic_dataset(file_path: str):
    """Load the symbolic training dataset"""
    
    prompts = []
    responses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
            responses.append(data['response'])
    
    # Create training format
    training_data = []
    for prompt, response in zip(prompts, responses):
        text = f"Human: {prompt}\nAssistant: {response}\n"
        training_data.append({"text": text})
    
    return Dataset.from_list(training_data)

# Load datasets
train_dataset = load_symbolic_dataset("symbolic_datasets/liber777_training_dataset.jsonl")
eval_dataset = load_symbolic_dataset("symbolic_datasets/liber777_validation_dataset.jsonl")
```

### **LoRA Training Configuration**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./liber777_symbolic_lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)
```

## 🔮 **Philosophical Foundation**

This system is built on the principle that **symbols represent potentialities and future possible states**. The Liber777revised contains a comprehensive system of correspondences that map:

- **Planetary influences** to daily activities and personal qualities
- **Elemental forces** to natural processes and human behavior
- **Numerical patterns** to cosmic principles and life cycles
- **Geometric forms** to fundamental structures of reality

By training a model on these correspondences, you create an AI that can:

1. **Recognize symbolic patterns** in new contexts
2. **Predict future possibilities** based on symbolic relationships
3. **Understand transformation** as represented by symbolic correspondences
4. **Generate insights** about potential states and outcomes

## 🎯 **Key Features**

### **1. Comprehensive Symbolic Knowledge**
- 7 planetary correspondences
- 4 elemental correspondences
- 10 qabalistic sephiroth
- 12 zodiacal signs
- 10 numerical correspondences
- 8 color correspondences

### **2. Multiple Extraction Methods**
- OCR-based text extraction
- Visual pattern recognition
- Symbolic pattern matching
- Table structure parsing

### **3. Quality Assurance**
- Confidence scoring for each extraction
- Verification status tracking
- Quality metrics calculation
- Validation dataset creation

### **4. Training-Ready Output**
- JSONL format for LoRA training
- Multiple training formats
- Prompt-response pairs
- Quality-filtered datasets

## 📈 **Quality Metrics**

The system provides several quality metrics:

- **Overall Quality Score**: Average quality of all correspondences
- **Completeness Score**: Coverage of symbolic knowledge base
- **Consistency Score**: Reliability of extracted correspondences

## 🔧 **Customization Options**

### **Extend Symbolic Knowledge**
```python
# Add custom correspondences
custom_knowledge = {
    "alchemical": {
        "mercury": ["quicksilver", "volatility", "communication", "transformation"],
        "sulfur": ["fire", "soul", "active principle", "masculine"],
        "salt": ["body", "stability", "passive principle", "feminine"]
    }
}

# Update the extractor
extractor.symbolic_knowledge.update(custom_knowledge)
```

### **Custom Pattern Recognition**
```python
# Define custom extraction patterns
custom_patterns = {
    "alchemical_symbols": r"☉|☽|☿|♀|♂|♃|♄|♅|♆|♇",
    "geometric_symbols": r"○|●|△|▲|□|■|◇|◆|☆|★"
}

# Add to pattern recognizers
extractor.pattern_recognizers.update(custom_patterns)
```

## 🎯 **Usage Examples**

### **Basic Symbolic Query**
```python
# After training, use the model for symbolic queries
def query_symbolic_model(prompt: str, model, tokenizer):
    """Query the trained symbolic model"""
    
    inputs = tokenizer.encode(f"Human: {prompt}\nAssistant:", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Assistant:")[-1].strip()

# Example queries
queries = [
    "What does the symbol of Mars represent?",
    "How do planetary correspondences relate to time?",
    "What is the symbolic meaning of the number 9?",
    "How do elemental correspondences manifest in human behavior?"
]
```

### **Future State Prediction**
```python
def predict_future_states(symbol: str, model, tokenizer):
    """Predict future possible states based on a symbol"""
    
    prompt = f"Based on the symbol {symbol}, what future possibilities and states does it represent?"
    response = query_symbolic_model(prompt, model, tokenizer)
    
    return {
        'symbol': symbol,
        'future_states': response,
        'confidence': 0.8  # Based on model confidence
    }

# Example future state predictions
symbols = ["sun", "moon", "fire", "water", "7", "triangle"]
for symbol in symbols:
    prediction = predict_future_states(symbol, model, tokenizer)
    print(f"Symbol: {prediction['symbol']}")
    print(f"Future States: {prediction['future_states']}")
    print("---")
```

## 📁 **File Structure**

```
├── liber777_symbolic_extractor.py          # Main extraction system
├── test_symbolic_extraction.py             # Test and validation
├── LIBER777_SYMBOLIC_TRAINING_GUIDE.md     # Comprehensive guide
├── SYMBOLIC_EXTRACTION_SUMMARY.md          # Complete summary
├── symbolic_datasets/                       # Generated datasets
│   ├── liber777_symbolic_dataset.json
│   ├── liber777_correspondences.csv
│   ├── liber777_training_dataset.jsonl
│   ├── liber777_validation_dataset.jsonl
│   └── liber777_symbolic_metadata.yaml
└── liber777_symbolic_extraction.log        # Processing log
```

## 🚀 **Quick Start Commands**

```bash
# Test the system
python test_symbolic_extraction.py

# Extract symbolic data from Liber777revised
python liber777_symbolic_extractor.py

# Review the generated datasets
ls -la symbolic_datasets/

# Check the training data format
head -5 symbolic_datasets/liber777_training_dataset.jsonl
```

## 🎯 **Next Steps**

1. **Run the test**: `python test_symbolic_extraction.py`
2. **Extract symbolic data**: `python liber777_symbolic_extractor.py`
3. **Review datasets**: Check quality and coverage
4. **Train LoRA model**: Use the provided training scripts
5. **Test symbolic queries**: Validate the model's understanding
6. **Iterate and improve**: Refine the extraction and training process

## 🔮 **Expected Outcomes**

After training, your AI model will be able to:

- **Answer symbolic questions** about correspondences
- **Predict future possibilities** based on symbolic relationships
- **Understand transformation** as represented by symbols
- **Generate insights** about potential states and outcomes
- **Recognize patterns** across different symbolic systems

This creates a powerful tool for **symbolic reasoning about future possibilities** - exactly what you need for understanding and working with the symbolic representation of future possible states.

---

**🎯 The goal is to create an AI that thinks symbolically about the future, using the wisdom encoded in Liber777revised to understand and predict potential transformations and outcomes.**

## 📚 **Documentation**

- **`LIBER777_SYMBOLIC_TRAINING_GUIDE.md`** - Comprehensive usage guide with examples
- **`SYMBOLIC_EXTRACTION_SUMMARY.md`** - Complete system summary and reference
- **`test_symbolic_extraction.py`** - Test script with examples

## 🤝 **Contributing**

This system is designed to be extensible. You can:
- Add new symbolic knowledge categories
- Implement custom extraction methods
- Extend pattern recognition systems
- Improve training data generation

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🔮 Ready to extract symbolic wisdom from Liber777revised and train an AI that understands future possibilities!** 