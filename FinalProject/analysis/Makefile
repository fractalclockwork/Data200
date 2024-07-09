TOP_DIR = ../..

all:
	@make lint
	@make test

lint:
	@black -l 75 *.py 
	@black -l 75 *.ipynb
	@flake8 --ignore=E501,E345,E722,E731,W503 *.py

environment:
	$(MAKE) -C $(TOP_DIR) environment

#test: 
#	pytest -v 

clean:
	rm -rf __pycache__

cleanall:
	@make clean
