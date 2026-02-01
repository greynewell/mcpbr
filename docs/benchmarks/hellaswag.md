---
description: "HellaSwag benchmark for evaluating commonsense reasoning through adversarially filtered sentence completion."
benchmark_howto:
  name: "HellaSwag"
  description: "Tests commonsense reasoning by asking models to choose the most plausible continuation of a scenario from four adversarially filtered options."
  benchmark_id: "hellaswag"
faq:
  - q: "What does HellaSwag measure?"
    a: "HellaSwag measures commonsense reasoning ability by presenting a scenario and four possible continuations. The model must select the most plausible option. The dataset is adversarially filtered so that incorrect options are deceptively plausible to language models while easy for humans (~95% accuracy)."
  - q: "How should the model format its answer for HellaSwag?"
    a: "The model should respond with a single digit (0, 1, 2, or 3) corresponding to the chosen option. The evaluation extracts the last occurrence of a digit in the 0-3 range from the response using regex matching."
  - q: "Can I filter HellaSwag tasks by topic?"
    a: "Yes, use the filter_category parameter with activity labels such as 'cooking', 'cleaning', 'sports', etc. Activity labels are matched case-insensitively."
---

# HellaSwag

## Overview

| Property | Value |
|----------|-------|
| **Benchmark ID** | `hellaswag` |
| **Dataset** | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) |
| **Tasks** | ~10,000 validation examples |
| **Evaluation** | Exact match of selected option (0-3) against correct label |
| **Output Type** | Single digit (0-3) |
| **Timeout** | 60-180 seconds |

!!! tip "Quick Start"
    ```bash
    mcpbr run -c config.yaml --benchmark hellaswag -n 20
    ```

## Overview

HellaSwag (Harder Endings, Longer contexts, and Low-shot Activities for Situations With Adversarial Generations) is a commonsense reasoning benchmark where the model must choose the most plausible continuation of a scenario from four options. The dataset was created through adversarial filtering, meaning the incorrect options are specifically designed to fool language models while remaining easy for humans to distinguish.

Human accuracy on HellaSwag is approximately 95%, making it a strong test of whether language models truly understand commonsense physical and social situations rather than relying on surface-level statistical patterns.

Key characteristics of HellaSwag:

- **Adversarial filtering**: Incorrect options are generated to be maximally confusing to models
- **Grounded scenarios**: Tasks are based on real-world activities from WikiHow and ActivityNet
- **Activity-labeled**: Each scenario is tagged with an activity label (e.g., "cooking", "cleaning")
- **Four-way multiple choice**: Exactly one correct answer among four options

## Task Structure

Each HellaSwag task contains the following fields:

- **ctx** (context): The scenario setup describing an ongoing situation
- **endings**: A list of exactly 4 possible continuations (indexed 0-3)
- **label**: The index of the correct continuation (0, 1, 2, or 3)
- **activity_label**: The category of activity (e.g., "Baking cookies", "Playing basketball")
- **ind**: The unique identifier for the task within the dataset

The agent receives the context and all four options, then must select the most plausible continuation by responding with the corresponding number.

### Example Task

```text
Scenario: A woman is seen sitting at a table with a bowl in front of her.
She picks up a whisk and begins stirring the contents of the bowl.

Options:
  (0) She then places the bowl in the oven and waits.
  (1) She adds flour to the bowl and continues mixing until smooth.
  (2) She throws the bowl across the room and walks away.
  (3) She picks up a phone and makes a call while stirring.

Correct Answer: 1
```

## Running the Benchmark

=== "CLI"

    ```bash
    # Run HellaSwag with default settings
    mcpbr run -c config.yaml --benchmark hellaswag

    # Run a small sample
    mcpbr run -c config.yaml --benchmark hellaswag -n 20

    # Filter by activity type
    mcpbr run -c config.yaml --benchmark hellaswag --filter-category "baking cookies"

    # Run with verbose output and save results
    mcpbr run -c config.yaml --benchmark hellaswag -n 100 -v -o results.json
    ```

=== "YAML"

    ```yaml
    benchmark: "hellaswag"
    sample_size: 10
    timeout_seconds: 120

    # Optional: filter to specific activity types
    filter_category:
      - "baking cookies"
      - "playing basketball"
    ```

### Activity Label Filtering

HellaSwag supports filtering by activity label using `filter_category`. Activity labels describe the type of scenario and come from WikiHow article titles and ActivityNet captions. Examples include:

- Cooking and baking activities
- Sports and physical activities
- Cleaning and household tasks
- Personal grooming and hygiene
- Social interactions
- Outdoor activities

Activity label matching is case-insensitive.

## Evaluation Methodology

HellaSwag evaluation follows a straightforward approach:

1. **Answer extraction**: The evaluation uses a regex pattern `\b([0-3])\b` to find all standalone digits in the range 0-3 within the model's response.

2. **Last-match selection**: The **last** matching digit in the response is used as the model's answer. This handles cases where the model discusses multiple options before stating its final choice.

3. **Exact comparison**: The extracted digit is compared to the ground truth `label` field. The task is resolved if and only if they match exactly.

### Scoring

```
resolved = (extracted_answer == correct_label)
```

Where:
- `extracted_answer`: Last digit in [0-3] found via regex in the model's response
- `correct_label`: The ground truth label from the dataset (as a string)

### Answer Format

The model should respond with a single digit. The evaluation is forgiving about surrounding text -- it will extract the answer from responses like:

- `"1"` (direct answer)
- `"The answer is 1."` (sentence format)
- `"After considering all options, I believe option 1 is most plausible."` (reasoned answer)
- `"Option 0 seems unlikely... Option 2 is too extreme... I'll go with 1."` (elimination reasoning -- uses last digit)

## Example Output

### Successful Evaluation

```json
{
  "resolved": true,
  "agent_answer": "1",
  "correct_label": "1"
}
```

### Failed Evaluation (Wrong Answer)

```json
{
  "resolved": false,
  "agent_answer": "3",
  "correct_label": "1"
}
```

### Failed Evaluation (No Answer Extracted)

```json
{
  "resolved": false,
  "error": "Could not extract option number from solution"
}
```

This occurs when the model's response does not contain any standalone digit in the range 0-3.

## Troubleshooting

### Model provides reasoning but no clear answer

If the model gives a long explanation without a clear numeric answer, the regex extraction may fail or pick the wrong number. Configure your agent prompt to explicitly request a single digit:

```yaml
agent_prompt: |
  {problem_statement}

  Respond with ONLY the number of the correct option (0, 1, 2, or 3). Do not include any explanation.
```

### Evaluation picks the wrong number from the response

The evaluation uses the **last** digit (0-3) found in the response. If the model discusses multiple options (e.g., "Option 0 is unlikely, option 2 is too extreme, so 1 is best"), it correctly selects `1`. However, if the model says "I choose 1, as options 0 and 3 are wrong", it would incorrectly select `3`. Instruct the model to state its final answer last.

### Low accuracy despite seemingly reasonable answers

HellaSwag was adversarially filtered specifically to trick language models. The incorrect options are designed to be statistically plausible even though they are wrong from a commonsense perspective. This is by design and measures genuine commonsense reasoning rather than pattern matching.

### Filter category returns no results

Activity labels in HellaSwag can be quite specific (e.g., "Baking cookies" rather than just "cooking"). Inspect the dataset to find exact labels:

```bash
uv run python -c "
from datasets import load_dataset
ds = load_dataset('Rowan/hellaswag', split='validation')
labels = sorted(set(item['activity_label'] for item in ds))
for label in labels[:20]:
    print(label)
print(f'... ({len(labels)} total labels)')
"
```

## Best Practices

- **Use direct answer prompts**: Since evaluation relies on extracting a single digit, instruct the model to respond with just the number. Verbose reasoning increases the risk of incorrect extraction.
- **Start with small samples**: Run 10-20 tasks first to verify the model's response format is compatible with the answer extraction logic.
- **Activity-specific evaluation**: Use `filter_category` to evaluate performance on specific types of commonsense reasoning. Some activities (e.g., physical tasks) may be harder than others (e.g., social situations).
- **Keep timeouts short**: HellaSwag tasks are simple multiple-choice questions. A timeout of 60-120 seconds is typically sufficient.
- **Use as a baseline benchmark**: HellaSwag scores provide a quick measure of commonsense reasoning that can be compared across models before running more expensive benchmarks.
- **Monitor answer extraction**: Check the `agent_answer` field in results to verify the extraction is picking up the intended answer from the model's response.

## Related Links

- [Benchmarks Overview](index.md)
- [TruthfulQA](truthfulqa.md) - Truthfulness evaluation benchmark
- [ARC](arc.md) - Science question answering benchmark
- [HellaSwag Dataset](https://huggingface.co/datasets/Rowan/hellaswag)
- [HellaSwag Paper](https://arxiv.org/abs/1905.07830)
- [Configuration Reference](../configuration.md)
- [CLI Reference](../cli.md)
