SRC=$(wildcard content/*)

all: html

html: ${SRC}
	jupyter-book build .

pdf: _build/latex/book.pdf

_build/latex/book.pdf: ${SRC} prepare_tex.sh post_process_tex.py
	source prepare_tex.sh
	jupyter-book build . --builder latex --toc _toc_tex.yml
	python post_process_tex.py ./_build/latex/book.tex
	cd ./_build/latex/ && make && cd -

clean:
	rm -fR _build/
