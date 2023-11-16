import sys
import re

target_filename = sys.argv[1]
fp = open(target_filename, "r", encoding="utf8")
content = fp.read()
fp.close()

edited_content = re.sub(r"(?s)\\begin\{sphinxtheindex\}.*\\end\{sphinxtheindex\}", "", content)
edited_content = edited_content.replace(r"\noindent\sphinxincludegraphics", 
                                        r"\noindent\centering\sphinxincludegraphics")
edited_content = edited_content.replace(r"\section{References}", 
                                        r"% \section{References}")
edited_content = edited_content.replace(r"\date{", 
                                        r"% \date{")


fp = open(target_filename, "w", encoding="utf8")
fp.write(edited_content)
fp.close()
