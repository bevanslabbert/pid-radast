VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PATH := src

.PHONY: venv install run clean

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: venv
	$(PYTHON) $(PATH)/main.py

clean:
	rm -rf $(VENV)
