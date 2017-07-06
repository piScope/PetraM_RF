PYTHON=$(shell which python)

INSTALLDIR=$(shell echo $(PetraM))
ifeq ($(INSTALLDIR),)
   INSTALLDIR := /usr/local/PetraM
endif
BINDIR=$(PETRAMDIR)/bin
LIBDIR=$(PETRAMDIR))/lib

default: compile

compile:
	$(PYTHON) setup.py build
install:
	$(PYTHON) setup.py install --prefix=$(INSTALLDIR)
clean:
	$(PYTHON) setup.py clean
