for lang in en fr;
do
    mkdir -p content_tex/${lang}
    for fname in content/${lang}/*.md; 
    do
        sed "s/%config InlineBackend.figure_format = 'svg'//g" ${fname} | sed "s/tex_specific_width:/width:/" | sed "s/is_html_output = True/is_html_output = False/" > ${fname/content/content_tex};
    done
    cp content/${lang}/*.py content_tex/${lang}/;

    cp -R content/data content_tex/;
    cp -R content/img content_tex/;
    cp content/references.bib content_tex/${lang}/; #Probably useless

    sed "s/content\//content_tex\//g" _toc_${lang}.yml > _toc_tex_${lang}.yml;
done
