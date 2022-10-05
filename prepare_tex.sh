mkdir -p content_tex/
for fname in content/*.md; 
do
    sed "s/%config InlineBackend.figure_format = 'svg'//g" ${fname} > ${fname/content/content_tex};
done

cp -R content/data content_tex/
cp content/references.bib content_tex/