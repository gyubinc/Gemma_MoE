# Qwen-MoE Dataset Documentation

## Overview

- **Total Domains**: 3
- **Successful**: 3
- **Failed**: 0
- **Total Samples**: 282,952

## Domain Details

### Medical Domain

- **Dataset**: medmcqa
- **Format**: multiple_choice_qa
- **Language**: english
- **Total Samples**: 193,155
- **Train**: 182,822
- **Validation**: 4,183
- **Test**: 6,150

**JSON Structure**:
  - `id`: str
  - `question`: str
  - `options`: list
  - `correct_option`: int
  - `correct_answer`: str
  - `formatted_question`: str
  - `formatted_answer`: str
  - `subject`: str
  - `topic`: str
  - `explanation`: str

### Law Domain

- **Dataset**: lex_glue_case_hold
- **Format**: legal_case_analysis
- **Language**: english
- **Total Samples**: 52,500
- **Train**: 45,000
- **Validation**: 3,900
- **Test**: 3,600

**JSON Structure**:
  - `context`: str
  - `endings`: list
  - `correct_ending_idx`: int
  - `correct_ending`: str
  - `formatted_question`: str
  - `formatted_answer`: str

### Math Domain

- **Dataset**: mathqa
- **Format**: multiple_choice_qa
- **Language**: english
- **Total Samples**: 37,297
- **Train**: 29,837
- **Validation**: 4,475
- **Test**: 2,985

**JSON Structure**:
  - `id`: str
  - `question`: str
  - `options`: list
  - `correct_option`: int
  - `correct_answer`: str
  - `formatted_question`: str
  - `formatted_answer`: str
  - `explanation`: str
  - `category`: str
  - `annotated_formula`: str
  - `linear_formula`: str

## Recommendations

- 💡 Consider standardizing formats across domains. Found formats: {'legal_case_analysis', 'multiple_choice_qa'}
- 💡 Consider balancing sample sizes across domains for better training

## Data Format Standards

### Required Fields in summary.json
- `dataset_name`: Name of the dataset
- `domain`: Domain category (medical, law, math)
- `total_samples`: Total number of samples
- `train_samples`: Number of training samples
- `validation_samples`: Number of validation samples
- `test_samples`: Number of test samples
- `format`: Data format type
- `language`: Language of the data

### Multiple Choice Format
For multiple choice questions (medical, law, math), each sample should contain:
- `question` or `context`: The question text
- `options` or `endings`: List of possible answers
- `correct_option` or `correct_ending_idx`: Index of correct answer
- `correct_answer`: Text of correct answer
- `formatted_question`: Formatted question with options
- `formatted_answer`: Formatted answer
