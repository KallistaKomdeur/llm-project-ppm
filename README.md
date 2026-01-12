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

## Running the project
To start the main program, run the following in the terminal:
```
python -m test_llm <log_name> <provider> <configuration> <OPTIONAL: --model>
```
The input parameters are:
1. log_name: name of the event log. Can be any of "your_event_log"
2. provider: name of the LLM provider. Restricted to "gemini", "openai", "anthropic"
3. configuration: which input configuration to use. Restricted to "single", "global_only", "inter-case_only"
4. OPTIONAL --model: which model to use from the LLM provider. Default to gemini 2.5-flash

If you want to only see the prompt and not the result, change in "test_llm.py" flag "print_only" to True. 
