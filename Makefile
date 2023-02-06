SRC_EN=$(wildcard content/en/*)
SRC_FR=$(wildcard content/fr/*)

all: html_en

html: html_en html_fr

html_en: ${SRC_EN}
	jupyter-book build . --path-output _build/html/en/ --toc _toc_en.yml

html_fr: ${SRC_FR}
	jupyter-book build . --path-output _build/html/fr/ --toc _toc_fr.yml

pdf: pdf_en pdf_fr

pdf_en: _build/latex/en/book.pdf

pdf_fr: _build/latex/fr/book.pdf

_build/latex/en/book.pdf: prepare_tex.sh post_process_tex.py ${SRC_EN}
	source prepare_tex.sh
	jupyter-book build . --builder latex --toc _toc_tex_en.yml --path-output _build/latex/en/
	python post_process_tex.py ./_build/latex/en/_build/latex/book.tex
	cd ./_build/latex/en/ && make && cd -
	mv ./_build/latex/en/_build/latex/book.pdf ./_build/latex/en/book.pdf

_build/latex/fr/book.pdf: prepare_tex.sh post_process_tex.py ${SRC_FR}
	source prepare_tex.sh
	jupyter-book build . --builder latex --toc _toc_tex_fr.yml --path-output _build/latex/fr/
	python post_process_tex.py ./_build/latex/fr/_build/latex/book.tex
	cd ./_build/latex/fr/_build/latex/ && make && cd -
	mv ./_build/latex/fr/_build/latex/book.pdf ./_build/latex/fr/book.pdf

clean:
	rm -fR _build/
