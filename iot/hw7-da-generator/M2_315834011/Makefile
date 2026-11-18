

REQUIRE = $(wildcard libraries/*/requirements.txt)
RPI_SUDO_PYTHON = sudo $(printenv VIRTUAL_ENV)/bin/python3

run: $(REQUIRE)
	python -m iottalkpy.dai sa.py

run_sudo: $(REQUIRE)
	sudo $(VIRTUAL_ENV)/bin/python3 -m iottalkpy.dai sa.py

pip_libs: $(REQUIRE)
	$(foreach var,$(REQUIRE),pip install -r $(var);)

pip_setup: pip_libs
	pip install -r requirements.txt

rpi_pip_setup: pip_setup
	pip install RPi.GPIO
