mkdir -p content_tex/
for fname in content/*.md; 
do
    sed "s/%config InlineBackend.figure_format = 'svg'//g" ${fname} | sed "s/tex_specific_width:/width:/" | sed "s/is_html_output = True/is_html_output = False/" > ${fname/content/content_tex};
done
cp content/*.py content_tex/

cp -R content/data content_tex/
cp content/references.bib content_tex/

sed "s/content\//content_tex\//g" _toc.yml > _toc_tex.yml;
