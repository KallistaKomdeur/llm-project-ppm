# LLM project

## Installation

### 1. Make sure Python is installed

This project requires Python 3.9+. You can check your version by running the following in the terminal:

```bash
python --version
```

If Python is not installed, download it from: https://www.python.org/downloads/.

### 2. Install dependencies

To install the required dependencies, run the following in the terminal:

```bash
pip install -r requirements.txt
```

### 3. Prepare your log

Since logs are too large to store on GitHub, you'll have to upload yours yourself. To do so:

1. Make a folder "logs" in the root directory.
2. In "logs", make a folder with the name of your event log
3. In "logs/your_event_log", place file "your_event_log.csv"

### 4. Prepare your environment.

In the project root, add a .env file. This file is ignored (see .gitignore), so your keys won't be pushed to github. In here, define:

- GEMINI_API_KEY
- OPENAI_API_KEY
- ANTHROPIC_API_KEY

## Running the project

### Getting LLM responses

To start the main program, run the following in the terminal:

```
python -m test_llm <log_name> <provider> <configuration> <OPTIONAL: --model>
```

The input parameters are:

1. log_name: name of the event log. Can be any of "your_event_log"
2. provider: name of the LLM provider. Restricted to "gemini", "openai", "anthropic"
3. configuration: which input configuration to use. Restricted to "single_x", "inter-case_x", "single_split_x", "inter-case_split_x, where x is any value for which there exists a prompt.
4. OPTIONAL --model: which model to use from the LLM provider. Default to gemini 2.5-flash, other options gpt-4o-mini

If you want to only see the prompt and not the result, change in "test_llm.py" flag "print_only" to True.

### Evaluating LLM responses

To evaluate LLM results for a particular log, configuration, or provider, run the following in the terminal:

```
python -m evaluate <log_name> <provider> <configuration>
```

The input parameters are the same as those described in section "Getting LLM responses".

### Benchmark evaluation

XGBoost was used as benchmark. To train and tune an XGBoost model on the data (both raw and cleaned), run the following in the terminal:

```
python train_xgboost.py <log_name>
```
