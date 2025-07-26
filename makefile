VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PATH := src

.PHONY: venv install run clean

venv:
	source ./$(VENV)/bin/activate

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: venv
	$(PYTHON) $(PATH)/main.py

clean:
	rm -rf $(VENV)
