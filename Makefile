SHELL := /bin/bash

SRC_EN=$(wildcard content/en/*)
SRC_FR=$(wildcard content/fr/*)

all: html pdf

html: html_en html_fr

html_en: ${SRC_EN}
	jupyter-book build . --path-output _build/html/en/ --toc _toc_en.yml --config _config_en.yml
	mv _build/html/en/_build/html _build/html/en_
	rm -fR _build/html/en
	mv _build/html/en_ _build/html/en
	echo '<meta http-equiv="Refresh" content="0; url=en/index.html" />' >> _build/html/index.html

html_fr: ${SRC_FR}
	jupyter-book build . --path-output _build/html/fr/ --toc _toc_fr.yml --config _config_fr.yml
	mv _build/html/fr/_build/html _build/html/fr_
	rm -fR _build/html/fr
	mv _build/html/fr_ _build/html/fr

pdf: pdf_en pdf_fr

pdf_en: _build/latex/book_en.pdf

pdf_fr: _build/latex/book_fr.pdf

_build/latex/book_en.pdf: prepare_tex.sh post_process_tex.py ${SRC_EN}
	chmod u+x prepare_tex.sh && ./prepare_tex.sh
	jupyter-book build . --builder latex --toc _toc_tex_en.yml --path-output _build/latex/en/ --config _config_en.yml
	python post_process_tex.py ./_build/latex/en/_build/latex/book.tex
	cd ./_build/latex/en/_build/latex/ && make && cd -
	mv ./_build/latex/en/_build/latex/book.pdf ./_build/latex/book_en.pdf

_build/latex/book_fr.pdf: prepare_tex.sh post_process_tex.py ${SRC_FR}
	chmod u+x prepare_tex.sh && ./prepare_tex.sh
	jupyter-book build . --builder latex --toc _toc_tex_fr.yml --path-output _build/latex/fr/ --config _config_fr.yml
	python post_process_tex.py ./_build/latex/fr/_build/latex/book.tex
	cd ./_build/latex/fr/_build/latex/ && make && cd -
	mv ./_build/latex/fr/_build/latex/book.pdf ./_build/latex/book_fr.pdf

clean:
	rm -fR _build/
