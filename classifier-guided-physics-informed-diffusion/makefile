# ==========================
# CHPC LENGAU HELPER MAKEFILE
# ==========================

SSH_USER = bslabbert
SSH_HOST = lengau.chpc.ac.za

# Default target
all:
	@echo "Available commands:"
	@echo "  make ssh             -> Connect to CHPC Lengau"
	@echo "  make nodes           -> View node info"
	@echo "  make modules         -> List available modules (paged)"
	@echo "  make search term=<module> -> Search for module"
	@echo "  make load module=<module> -> Load a specific module"
	@echo "  make list            -> Show loaded modules"
	@echo "  make job             -> Request an interactive job"
	@echo "  make htop            -> Run htop on cluster"

# SSH into Lengau
ssh:
	ssh $(SSH_USER)@$(SSH_HOST)

# List all nodes
nodes:
	pbsnodes -a | less

# Show all available modules
modules:
	module avail 2>&1 | less

# Search available modules (usage: make search term=python)
search:
	@if [ -z "$(term)" ]; then \
		echo "Usage: make search term=<search_term>"; \
	else \
		module avail 2>&1 | grep $(term); \
	fi

# Load a module (usage: make load module=python/3.9)
load:
	@if [ -z "$(module)" ]; then \
		echo "Usage: make load module=<module_name>"; \
	else \
		module load $(module); \
	fi

# List loaded modules
list:
	module list

# Submit interactive job
job:
	qsub -X -I -l select=1:ncpus=6:mpiprocs=6 -q serial -P CSCI1166 -l walltime=2:00:00

# Run htop inside job
htop:
	htop
