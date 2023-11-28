Search.setIndex({"docnames": ["content/fr/convnets", "content/fr/intro", "content/fr/loss", "content/fr/mlp", "content/fr/optim", "content/fr/perceptron", "content/fr/regularization", "content/fr/rnn"], "filenames": ["content/fr/convnets.md", "content/fr/intro.md", "content/fr/loss.md", "content/fr/mlp.md", "content/fr/optim.md", "content/fr/perceptron.md", "content/fr/regularization.md", "content/fr/rnn.md"], "titles": ["R\u00e9seaux neuronaux convolutifs", "Introduction au Deep Learning", "Fonctions de co\u00fbt", "Perceptrons multicouches", "Optimisation", "Introduction", "R\u00e9gularisation", "R\u00e9seaux neuronaux r\u00e9currents"], "terms": {"Les": [0, 1, 4, 5, 6], "auss": [0, 2, 3, 4, 5, 6, 7], "appel": [0, 2, 3, 4, 5, 6, 7], "convnet": 0, "con\u00e7us": [0, 7], "tir": [0, 4, 6], "part": [0, 3, 4, 5, 6, 7], "structur": [0, 3], "don": [0, 1, 2, 3, 5, 6, 7], "dan": [0, 2, 3, 4, 5, 6], "chapitr": [0, 3, 4, 5, 6, 7], "abord": [0, 3, 4, 7], "deux": [0, 2, 4, 6, 7], "commenc": [0, 5], "cas": [0, 1, 2, 4, 5, 7], "monodimensionnel": 0, "verron": 0, "comment": [0, 4, 5], "1d": 0, "peuvent": [0, 4, 6], "\u00eatre": [0, 3, 4, 5, 6, 7], "util": [0, 4, 6], "trait": [0, 1, 4, 7], "Nous": [0, 2, 3, 5, 7], "pr\u00e9sent": [0, 2, 3, 4, 5, 6, 7], "ensuit": 0, "2d": 0, "particuli": 0, "reposent": [0, 7], "oper": 0, "mathbf": [0, 4, 5], "x": [0, 2, 3, 4, 5, 6, 7], "filtr": [0, 7], "f": [0, 4], "calcul": [0, 3, 4, 5, 6, 7], "cart": 0, "activ": [0, 4, 5, 6, 7], "comm": [0, 1, 2, 3, 4, 5, 6, 7], "begin": [0, 2, 3, 4, 7], "equat": [0, 7], "left": [0, 3, 4, 5, 7], "right": [0, 3, 4, 5, 7], "sum_": [0, 2, 3, 4, 5, 6, 7], "k": [0, 6], "L": [0, 2, 3, 4, 5, 6], "f_": 0, "x_": [0, 3, 7], "label": [0, 3, 4, 6, 7], "eq": [0, 3], "conv1d": 0, "end": [0, 2, 3, 4, 7], "o\u00f9": [0, 2, 3, 4, 5, 6, 7], "longueur": 0, "2l": 0, "1": [0, 2, 4, 5, 7], "Le": [0, 1, 5, 6], "cod": [0, 3], "suiv": [0, 3, 4, 5, 6, 7], "illustr": [0, 4, 5, 6], "notion": [0, 1, 7], "utilis": [0, 1, 2, 3, 4, 5, 6, 7], "gaussien": 0, "config": [0, 2, 3, 4, 5, 6, 7], "inlinebackend": [0, 2, 3, 4, 5, 6, 7], "figure_format": [0, 2, 3, 4, 5, 6, 7], "svg": [0, 2, 3, 4, 5, 6, 7], "matplotlib": [0, 2, 3, 4, 5, 6, 7], "inlin": [0, 2, 3, 4, 5, 6, 7], "import": [0, 2, 3, 4, 5, 6, 7], "pyplot": [0, 2, 3, 4, 5, 6, 7], "plt": [0, 2, 3, 4, 5, 6, 7], "from": [0, 2, 3, 4, 5, 6, 7], "notebook_util": [0, 2, 3, 4, 5, 6, 7], "prepare_notebook_graphic": [0, 2, 3, 4, 5, 6, 7], "numpy": [0, 2, 3, 4, 5, 6, 7], "np": [0, 2, 3, 4, 5, 6], "def": [0, 3, 4, 5, 7], "random_walk": 0, "siz": [0, 4], "rnd": 0, "random": [0, 4], "randn": [0, 4], "ts": 0, "for": [0, 4, 5], "in": [0, 3, 4, 5, 6, 7], "rang": [0, 4, 5], "return": [0, 3, 4, 5, 7], "seed": [0, 4], "0": [0, 2, 3, 4, 5, 6, 7], "50": [0, 2, 3, 7], "exp": [0, 3, 4, 7], "linspac": [0, 2, 3, 4, 5, 7], "2": [0, 1, 2, 4, 5, 6, 7], "num": [0, 5], "5": [0, 3, 4, 5, 6], "sum": [0, 4, 5], "figur": [0, 3, 4, 5, 6, 7], "plot": [0, 2, 3, 4, 5, 6, 7], "correlat": 0, "mod": 0, "sam": 0, "pass": [0, 4, 7], "traver": [0, 7], "legend": [0, 4, 6, 7], "constitu": [0, 4, 5], "bloc": [0, 7], "dont": [0, 3, 5, 7], "parametr": [0, 2, 3, 4, 5, 6, 7], "coefficient": 0, "int\u00e8grent": 0, "donc": [0, 3, 5, 7], "fix": [0, 3, 5], "a": [0, 3, 4, 5, 6, 7], "prior": [0, 5], "exempl": [0, 3, 4, 6], "ci": [0, 3, 4, 6, 7], "dessus": [0, 3, 6, 7], "plut\u00f4t": [0, 3, 7], "appris": [0, 5], "Ces": [0, 4], "\u00e9quivari": 0, "translat": [0, 7], "signif": [0, 4, 7], "d\u00e9calag": 0, "entr\u00e9": [0, 3, 4, 5, 6, 7], "entra\u00een": [0, 2, 4, 6, 7], "m\u00eam": [0, 3, 4, 6, 7], "sort": [0, 2, 4, 5, 6, 7], "ipython": [0, 4], "display": [0, 4, 6], "html": [0, 4, 6], "celluloid": 0, "cam": 0, "zeros": [0, 4], "12": [0, 3, 4], "4": [0, 2, 3, 4, 5, 6, 7], "8": [0, 3, 4, 7], "length": [0, 3], "60": [0, 3], "fig": [0, 4], "pos": [0, 3], "list": 0, "35": 0, "100": [0, 3, 4, 5, 6], "sin": 0, "pi": 0, "act": 0, "subplot": [0, 3, 4], "b": [0, 3, 4, 5, 7], "titl": [0, 3, 4], "input": [0, 4, 7], "tim": [0, 3], "fig2": 0, "r": [0, 4, 5, 6, 7], "map": 0, "axes2": 0, "add_ax": 0, "15": [0, 6], "renvoi": 0, "objet": 0, "axe": [0, 4, 7], "set_xtick": 0, "set_titl": [0, 4], "filt": 0, "tight_layout": [0, 4], "snap": 0, "anim": [0, 4], "animat": [0, 4], "clos": [0, 4], "to_jshtml": [0, 4], "tmp": 0, "ipykernel_8213": 0, "368849627": 0, "py": 0, "32": 0, "userwarning": 0, "this": 0, "includ": 0, "that": 0, "are": 0, "not": [0, 1, 3, 6, 7], "compatibl": 0, "with": [0, 4, 7], "so": 0, "result": 0, "might": 0, "be": 0, "incorrect": 0, "once": [0, 4], "loop": [0, 4], "reflect": [0, 4], "model": [0, 2, 3, 4, 6, 7], "connus": [0, 7], "tres": [0, 3, 4, 5, 6, 7], "perform": [0, 4, 6], "appliqu": [0, 3, 6, 7], "vision": 0, "ordin": 0, "quantit": [0, 3, 4, 5, 7], "moder": 0, "rapport": [0, 4, 5, 7], "entier": [0, 3], "connect": [0, 3], "bien": 0, "s\u00fbr": 0, "contr": 0, "existent": 0, "term": [0, 3, 4, 5, 6], "vagu": 0, "La": [0, 2, 3, 6, 7], "plupart": 0, "architectur": [0, 5, 7], "standard": [0, 2, 4], "convolutionnel": [0, 5], "adapt": [0, 4, 5, 7], "direct": [0, 4, 5], "communaut": 0, "guennec": 0, "al": [0, 6, 7], "2016": 0, "appui": [0, 3, 4], "altern": [0, 4, 7], "entre": [0, 3, 4, 7], "tand": 0, "traval": [0, 1], "plus": [0, 2, 3, 4, 5, 6, 7], "r\u00e9cent": 0, "appuient": 0, "connex": [0, 6], "r\u00e9siduel": 0, "modul": 0, "incept": 0, "fawaz": 0, "2020": 0, "bas": [0, 1, 3, 4, 5, 7], "discut": [0, 3, 5], "d\u00e9tail": [0, 3, 5, 7], "section": [0, 2, 3, 4, 5, 6], "autr": [0, 3, 4, 5, 6, 7], "\u00e9valu": [0, 5, 6], "2019": [0, 7], "conseillon": 0, "lecteur": 0, "int\u00e9ress": [0, 3, 4], "allon": [0, 4, 5], "mainten": [0, 2, 3, 4, 6, 7], "lequel": [0, 4, 7], "gliss": 0, "seul": [0, 3, 4, 5, 6, 7], "dimens": 0, "largeur": 0, "hauteur": 0, "voit": 0, "dessous": [0, 3, 4, 6], "grill": 0, "pixel": 0, "chaqu": [0, 3, 4, 5, 6], "valeur": [0, 3, 4, 5, 6, 7], "intens": [0, 5], "chacun": [0, 3], "canal": 0, "couleur": 0, "typiqu": [0, 4, 6, 7], "compos": [0, 2, 3, 4], "3": [0, 4, 5, 6, 7], "roug": 0, "vert": 0, "bleu": 0, "imread": 0, "dat": [0, 3, 4, 5, 6], "cat": 0, "jpg": 0, "image_r": 0, "copy": [0, 3], "image_g": 0, "image_b": 0, "figsiz": [0, 3, 4], "20": [0, 3, 4, 5], "imshow": 0, "Une": [0, 2, 3, 6, 7], "rgb": [0, 3], "axis": [0, 4, 5, 6], "off": 0, "i": [0, 2, 3, 4, 5], "img": 0, "color": [0, 3, 4, 6], "enumerat": 0, "zip": [0, 4], "gauch": [0, 6], "droit": [0, 6], "nouvel": [0, 5, 7], "suit": [0, 2, 3, 4, 5, 6, 7], "conv2d": 0, "En": [0, 3, 4, 6, 7], "produit": [0, 7], "scalair": [0, 7], "tenseur": 0, "form": [0, 2, 3, 4, 7], "2k": 0, "patch": 0, "centr": [0, 4], "posit": [0, 3, 4], "consid\u00e9ron": [0, 4, 7], "9x9": 0, "sz": 0, "9": [0, 3, 4, 5, 7], "arang": [0, 4, 6], "reshap": [0, 4], "filter_3d": 0, "r\u00e9sultat": [0, 7], "chat": 0, "niveau": 0, "gris": [0, 6], "dir": [0, 4, 5], "scipy": [0, 4], "signal": 0, "convolve2d": 0, "convoluted_signal": 0, "shap": [0, 3, 4], "boundary": 0, "symm": 0, "cmap": 0, "gray": 0, "On": [0, 4, 7], "peut": [0, 2, 3, 4, 5, 6, 7], "remarqu": [0, 3, 4], "version": [0, 1, 7], "flou": 0, "original": [0, 7], "C": [0, 3, 4, 5, 6, 7], "parc": [0, 5], "avon": [0, 2, 3, 4, 5, 6, 7], "lor": [0, 3, 4, 6, 7], "contenu": [0, 3], "d\u00e9fin": [0, 2, 3, 4, 7], "lecun": [0, 4], "1998": 0, "empil": 0, "introduit": [0, 3, 5, 6, 7], "t\u00e2ch": [0, 5], "sp\u00e9cif": [0, 3, 5], "reconnaiss": 0, "chiffr": 0, "r\u00e9sult": [0, 4, 6], "repr\u00e9sent": [0, 3], "plusieur": [0, 3, 4, 6, 7], "\u00e9gal": [0, 3, 4, 5, 7], "kernel": 0, "op\u00e8rent": 0, "parallel": 0, "g\u00e9ner": [0, 5], "tout": [0, 1, 3, 6, 7], "tous": [0, 2, 3, 4, 6, 7], "partagent": [0, 3], "Un": [0, 3, 4, 6], "bi": [0, 2, 3, 4, 5, 7], "fonction": [0, 4, 5, 6, 7], "ensembl": [0, 2, 3, 4, 6], "varph": [0, 3, 4, 5], "prim": [0, 4, 7], "c_": [0, 7], "b_c": [0, 7], "conv_lai": 0, "d\u00e9sign": 0, "associ": [0, 3, 5, 7], "ker": [0, 1, 6], "tel": [0, 3, 4, 5, 6, 7], "impl\u00e9ment": [0, 6], "aid": [0, 6], "class": [0, 2, 3, 4], "keras_cor": [0, 3, 4, 6], "layer": [0, 3, 4, 6], "lai": [0, 3, 4], "filter": 0, "6": [0, 3, 4, 5], "kernel_siz": 0, "valid": [0, 6], "relu": [0, 3, 4, 6], "visualis": [0, 6], "effet": [0, 3, 4, 7], "sourc": 0, "v": [0, 7], "dumoulin": 0, "visin": 0, "guid": 0, "to": [0, 4, 6], "arithmetic": 0, "deep": [0, 5], "learning": [0, 4, 5, 6], "san": [0, 4, 5, 6], "assur": 0, "caract\u00e9rist": [0, 3, 7], "cel": [0, 3, 5, 6, 7], "r\u00e9alis": 0, "agrand": 0, "artificiel": 0, "rempl": 0, "zon": 0, "z\u00e9ros": 0, "blanc": 0, "effectuent": 0, "sous": [0, 4, 6, 7], "\u00e9chantillonnag": 0, "r\u00e9sum": [0, 7], "quelqu": [0, 7], "inform": [0, 3, 4, 5, 7], "faibl": [0, 5, 7], "r\u00e9solu": 0, "id\u00e9": [0, 4, 5, 6], "parcel": 0, "agr\u00e9gat": 0, "agreg": 0, "moyen": [0, 4, 5, 6], "correspond": [0, 3, 7], "averag": 0, "maximum": 0, "max": [0, 4], "afin": [0, 2, 3, 4, 6], "r\u00e9duir": [0, 6], "g\u00e9n\u00e9ral": [0, 3, 4, 7], "fen\u00eatr": 0, "chevauchent": 0, "taill": [0, 5], "2x2": 0, "larg": [0, 2, 3], "histor": [0, 3], "premi": [0, 2, 3, 7], "moin": [0, 4], "mesur": [0, 6], "puissanc": 0, "disponibl": [0, 4], "augment": 0, "maxpool2d": 0, "avgpool2d": 0, "max_pooling_lai": 0, "pool_siz": 0, "average_pooling_lai": 0, "prend": [0, 5, 7], "suppl\u00e9mentair": [0, 3, 4, 7], "diff\u00e9rent": [0, 3, 4, 6, 7], "lorsqu": [0, 3, 4, 6], "vis": [0, 7], "object": [0, 3, 5, 6], "produir": [0, 3], "probabl": [0, 2, 3], "head": 0, "Pour": [0, 3, 4, 5, 6, 7], "capabl": 0, "doivent": [0, 3, 4], "transform": [0, 7], "vecteur": [0, 5, 7], "Cette": [0, 3, 5, 6], "flatten": 0, "sequential": [0, 3, 4, 6], "inputlai": [0, 3, 4, 6], "dens": [0, 3, 4, 6], "input_shap": [0, 3, 4, 6], "16": [0, 3], "120": [0, 3, 5], "84": 0, "10": [0, 3, 4, 5, 6], "softmax": [0, 3, 4, 6], "summary": [0, 3, 4], "_________________________________________________________________": [0, 3, 4], "output": [0, 3, 4, 7], "param": [0, 3, 4], "non": [0, 3, 4, 5, 6], "28": 0, "156": 0, "max_pooling2d": 0, "maxpooling2": 0, "14": [0, 6], "D": [0, 2, 4, 6], "conv2d_1": 0, "2416": 0, "max_pooling2d_1": 0, "maxpoolin": 0, "g2d": 0, "400": 0, "48120": 0, "dense_1": [0, 3, 4], "10164": 0, "dense_2": 0, "850": 0, "total": [0, 3, 4], "61706": 0, "241": 0, "04": 0, "kb": [0, 3, 4], "trainabl": [0, 3, 4], "00": [0, 3, 4], "byt": [0, 3, 4], "ffw": 0, "19": 0, "hassan": 0, "ismail": 0, "germain": 0, "foresti": 0, "jonathan": 0, "web": 0, "lhassan": 0, "idoumghar": 0, "and": [0, 4, 6, 7], "pierr": 0, "alain": 0, "mull": 0, "review": 0, "mining": 0, "knowledg": 0, "discovery": 0, "33": [0, 3, 5], "917": 0, "963": 0, "flf": 0, "benjamin": 0, "luc": 0, "charlott": 0, "pelleti": 0, "daniel": 0, "schmidt": 0, "geoffrey": [0, 6], "webb": 0, "fran": 0, "\u00e7": 0, "ois": 0, "petitjean": 0, "inceptiontim": 0, "finding": 0, "alexnet": 0, "34": [0, 3, 5], "1936": 0, "1962": 0, "lgmt16": 0, "arthur": 0, "simon": 0, "malinowsk": 0, "romain": [0, 1], "tavenard": [0, 1], "using": [0, 3, 4, 6], "convolutional": 0, "neural": [0, 6, 7], "network": [0, 6], "ecml": 0, "pkdd": 0, "workshop": 0, "advanced": 0, "analytic": 0, "temporal": 0, "riv": 0, "del": 0, "gard": 0, "italy": 0, "septemb": 0, "lbbh98": 0, "yann": [0, 4], "\u00e9": 0, "bottou": 0, "yoshu": [0, 4, 7], "bengio": [0, 4, 7], "patrick": 0, "haffn": 0, "gradient": [0, 7], "based": 0, "applied": 0, "docu": [0, 1, 5], "recognit": 0, "proceeding": 0, "of": [0, 6, 7], "the": [0, 7], "iee": 0, "86": 0, "11": [0, 3, 4, 5], "2278": 0, "2324": 0, "Ce": [1, 5, 7], "sert": [1, 3], "cour": [1, 2, 4, 5, 6, 7], "dispens": 1, "univers": 1, "ren": 1, "franc": 1, "edhec": 1, "lill": 1, "r\u00e9seau": [1, 2, 3, 4, 5, 6], "neuron": [1, 3, 4, 5, 6, 7], "classif": [1, 2, 3, 4], "r\u00e9gress": [1, 2, 3, 5], "tabulair": 1, "compr": 1, "algorithm": [1, 3, 4, 5, 7], "optimis": [1, 2, 3, 6, 7], "perceptron": [1, 4], "multicouch": 1, "convolut": [1, 7], "imag": 1, "apprentissag": [1, 4, 5, 6], "transfert": 1, "pr\u00e9vis": 1, "s\u00e9quenc": [1, 7], "s\u00e9anc": 1, "pratiqu": [1, 3, 4], "nb": 1, "traduit": 1, "ver": [1, 3, 7], "mani": [1, 2, 3, 4, 6, 7], "sem": 1, "automat": [1, 6], "h\u00e9sit": 1, "r\u00e9fer": 1, "anglais": 1, "dout": 1, "famill": [2, 3, 5], "mlp": 2, "e": [2, 3, 4, 5, 6], "ajust": [2, 4], "adaptent": 2, "devon": 2, "loss": [2, 4, 5, 6], "function": 2, "fois": [2, 3, 4, 5], "chois": [2, 3, 5, 6], "consist": [2, 4, 6], "regl": [2, 4, 7], "minimis": [2, 5, 6], "savoir": [2, 3, 4], "principal": [2, 6, 7], "supposon": [2, 5], "connu": [2, 4, 6, 7], "mathcal": [2, 4, 5, 6], "\u00e9chantillon": [2, 4, 5], "annot": 2, "x_i": [2, 4, 5, 6], "y_i": [2, 4, 5, 6], "d\u00e9signon": [2, 6], "forall": [2, 3, 7], "hat": [2, 3, 4, 5], "_i": [2, 3, 4], "m_": [2, 4, 6], "thet": [2, 4, 6], "notr": [2, 3, 5], "poid": [2, 3, 4, 5, 6, 7], "mean": [2, 4, 6], "squared": 2, "error": 2, "mse": [2, 4], "context": [2, 3, 6], "Elle": [2, 4], "align": [2, 3, 4], "frac": [2, 3, 4, 5, 6, 7], "Sa": 2, "tend": [2, 6], "p\u00e9nalis": 2, "fort": [2, 4], "grid": [2, 3, 4, 7], "xlabel": [2, 4, 6], "ylabel": [2, 4, 6], "neuronal": [2, 3, 4, 5, 6], "log": [2, 3, 4, 5], "p": [2, 4], "pr\u00e9dit": [2, 3, 5], "correct": [2, 4], "formul": [2, 3, 4, 7], "favoris": 2, "pred": 2, "proch": 2, "attendr": [2, 6], "ion": 2, "01": [2, 6], "pr\u00e9c\u00e9dent": [3, 4, 5, 6, 7], "vu": [3, 4, 6], "simpl": [3, 5, 6], "combinaison": 3, "lin\u00e9air": [3, 5], "x_j": 3, "w_j": 3, "parm": 3, "assez": 3, "restreint": 3, "couvr": 3, "\u00e9ventail": 3, "organis": 3, "complex": 3, "cach": [3, 4, 6, 7], "car": [3, 4], "question": 3, "si": [3, 4, 5, 6, 7], "permet": [3, 7], "effect": 3, "grand": [3, 4, 5, 6, 7], "stipul": 3, "continu": [3, 6], "compact": 3, "approch": [3, 5], "pres": 3, "veut": 3, "sigmo\u00efd": [3, 7], "mettr": [3, 4], "propriet": 3, "cepend": [3, 6], "nombr": [3, 5, 6], "n\u00e9cessair": [3, 4], "obten": [3, 4], "qualit": 3, "De": 3, "suffis": [3, 5, 6], "bon": [3, 4, 5], "exist": [3, 7], "converg": [3, 5], "fin": 3, "garant": 3, "d\u00e9di": [3, 5], "observon": 3, "empir": [3, 6], "atteindr": 3, "efficac": 3, "requ": 3, "graphiqu": [3, 7], "varphi_": 3, "text": [3, 4, 5, 7], "out": 3, "w": [3, 4, 5, 7], "_": [3, 4, 5, 6], "teal": 3, "h": [3, 4, 6, 7], "sum_j": 3, "ij": 3, "61": 3, "91": 3, "blu": [3, 4], "mlp_2hidden": 3, "prec": 3, "mult": [3, 4, 5], "concept": [3, 5, 6], "destin": 3, "problem": [3, 4, 5], "certain": [3, 4, 5, 6], "hyp": [3, 4, 5], "prenon": 3, "c\u00e9lebr": 3, "jeu": [3, 4, 5, 6], "iris": [3, 4, 6], "pand": [3, 4, 5, 6], "pd": [3, 4, 5, 6], "read_csv": [3, 4, 5, 6], "csv": [3, 4, 5, 6], "index_col": [3, 4, 6], "sepal": 3, "cm": 3, "width": 3, "petal": 3, "target": [3, 4, 6], "7": [3, 4, 5], "145": 3, "146": 3, "147": [3, 5], "148": 3, "149": 3, "150": [3, 4], "row": [3, 5], "column": [3, 4, 5, 6], "apprendr": [3, 7], "d\u00e9duir": 3, "attribut": 3, "cibl": [3, 5], "possibl": 3, "dict": 3, "descript": 3, "puisqu": [3, 4], "cens": 3, "confront": 3, "situat": 3, "agit": [3, 6, 7], "pr\u00e9dir": [3, 5], "quand": 3, "binair": [3, 4], "aur": [3, 7], "indiqu": [3, 5], "aut": 3, "ains": 3, "restent": 3, "choix": [3, 5], "Il": [3, 4, 7], "utilison": [3, 6], "ident": [3, 5], "profondeur": 3, "r\u00e9gim": 3, "comportent": 3, "gamm": 3, "propos": 3, "tanh": [3, 4, 7], "2x": 3, "sigmoid": [3, 4], "gt": 3, "sinon": 3, "ylim": [3, 4], "vari": [3, 4, 7], "jour": [3, 4, 7], "raison": [3, 7], "consacr": 3, "Vous": [3, 5], "fourn": [3, 7], "\u00e9quat": 3, "possed": [3, 4], "propr": 3, "expliqu": [3, 6], "fait": [3, 4, 5, 6], "r\u00e9soudr": [3, 5], "pu": 3, "constat": [3, 4], "plag": [3, 4], "primordial": 3, "ad\u00e9quat": 3, "produis": 3, "coh\u00e9rent": 3, "boston": [3, 5], "parl": 3, "prix": [3, 5], "n\u00e9gat": 3, "ser": [3, 4, 5, 7], "judici": 3, "devr": [3, 5], "situ": 3, "intervall": 3, "alor": [3, 6, 7], "d\u00e9faut": 3, "enfin": [3, 7], "compris": [3, 7], "somm": [3, 4], "doit": [3, 7], "\u00c0": 3, "o_i": 3, "o_j": 3, "avant": [3, 4, 6], "suff": 3, "titr": 3, "unit": [3, 4, 6], "tensorflow": [3, 4, 6, 7], "backend": [3, 4, 6], "typ": [3, 4, 7], "220": [3, 4], "63": [3, 4], "283": [3, 4], "aper\u00e7u": 3, "pouv": [3, 5, 6], "retourn": [3, 7], "d\u00e9j\u00e0": 3, "200": 3, "connexion": 3, "ceux": 3, "Au": [3, 7], "but": [3, 5, 7], "pric": [3, 5], "rm": [3, 5], "crim": 3, "indus": 3, "nox": 3, "age": 3, "tax": 3, "575": [3, 5], "00632": 3, "31": 3, "538": 3, "65": 3, "296": 3, "24": [3, 5], "421": [3, 5], "02731": 3, "07": 3, "469": 3, "78": 3, "242": 3, "21": [3, 5], "185": [3, 5], "02729": 3, "998": [3, 5], "03237": 3, "18": 3, "458": 3, "45": 3, "222": 3, "06905": 3, "54": 3, "36": [3, 5], "501": [3, 5], "593": [3, 5], "06263": 3, "93": 3, "573": 3, "69": 3, "273": 3, "22": [3, 5], "502": [3, 5], "04527": 3, "76": 3, "503": [3, 5], "976": [3, 5], "06076": 3, "23": [3, 5], "504": [3, 5], "794": [3, 5], "10959": 3, "89": 3, "505": [3, 5], "030": [3, 5], "04741": 3, "80": 3, "506": [3, 5], "strateg": [4, 5, 6], "montr": 4, "elles": 4, "commen\u00e7on": 4, "limit": [4, 7], "initialis": 4, "pr\u00e9dict": [4, 5, 6], "individuel": 4, "nabla_": [4, 7], "mis": [4, 7], "iter": [4, 5], "leftarrow": 4, "rho": [4, 5], "m\u00e9thod": [4, 5], "taux": [4, 5], "rat": [4, 6], "diminu": 4, "pert": [4, 5], "voir": [4, 5, 6, 7], "epoch": [4, 6], "passag": 4, "complet": [4, 6, 7], "jeux": 4, "motiv": 4, "derri": [4, 6], "stochastic": 4, "sgd": 4, "estim": [4, 6], "march": 4, "sen": 4, "fair": [4, 5, 7], "minibatch": [4, 6], "produisent": 4, "apres": [4, 6], "dataset": 4, "n_": 4, "al\u00e9atoir": [4, 6], "tailll": 4, "Par": 4, "cons\u00e9quent": 4, "fr\u00e9quent": 4, "bruit": 4, "lieu": [4, 7], "vrai": [4, 6], "optimiz": [4, 6], "optim": 4, "grad": 4, "alpha": [4, 5], "lambd": [4, 6], "dot": [4, 7], "T": 4, "norm": 4, "sqrt": 4, "cost": 4, "todo": 4, "fass": 4, "nimp": 4, "optim_gd": 4, "alpha_in": 4, "n_epoch": [4, 6], "alphas": 4, "append": [4, 5], "concatenat": 4, "optim_sgd": 4, "minibatch_siz": 4, "scaled_lambd": 4, "indices_minibatch": 4, "randint": 4, "x_minibatch": 4, "y_minibatch": 4, "stretch_to_rang": 4, "lim": 4, "sz_rang": 4, "middl": 4, "get_lim": 4, "alphas_list": 4, "xlim": [4, 6], "min": [4, 6], "if": 4, "else": 4, "gen_anim": 4, "alphas_gd": 4, "alphas_sgd": 4, "alpha_star": 4, "n_steps_per_epoch": 4, "gen_video": 4, "tru": [4, 6], "global": [4, 6], "lines_alph": 4, "40": [4, 6], "nn": 4, "xv": 4, "yv": 4, "meshgrid": 4, "xvisu": 4, "ravel": 4, "pv": 4, "13": 4, "ax": 4, "contour": 4, "ko": [4, 5], "fillstyl": 4, "line_alph": 4, "mark": 4, "set_xlabel": 4, "w_0": [4, 5], "set_ylabel": 4, "w_1": 4, "set_xlim": 4, "set_ylim": 4, "text_epoch": 4, "set_xdat": 4, "set_ydat": 4, "set_text": 4, "ani": 4, "funcanim": 4, "interval": 4, "500": 4, "blit": 4, "fals": [4, 6], "save_count": 4, "len": [4, 6], "rand": 4, "astyp": 4, "int": 4, "2e": 4, "array": [4, 5], "res_optim": 4, "minimiz": 4, "fun": 4, "x0": 4, "jac": 4, "visualiz": 4, "is_html_output": 4, "viz": 4, "repeat": 4, "outr": [4, 5], "impliqu": 4, "avantag": 4, "essentiel": 4, "contrair": 4, "avion": [4, 6], "va": 4, "logist": 4, "convex": 4, "d\u00e8s": [4, 6], "celui": [4, 6], "couch": [4, 5, 6, 7], "model_forward_loss": 4, "weight": 4, "bias": 4, "0001": 4, "w0": 4, "75": 4, "wi": 4, "souffr": 4, "local": [4, 5, 7], "constituent": 4, "s\u00e9rieux": 4, "c\u00f4t": [4, 7], "susceptibl": 4, "b\u00e9n\u00e9fici": 4, "\u00e9chapp": 4, "minim": 4, "kingm": 4, "ba": 4, "2015": 4, "differ": 4, "momentum": 4, "ant\u00e9rieur": 4, "liss": 4, "trajectoir": 4, "espac": 4, "pend": [4, 6], "interact": 4, "trouv": [4, 5], "goh": 4, "2017": 4, "remplac": 4, "beta_1": 4, "z\u00e9ro": 4, "stock": [4, 5], "theta_": 4, "epsilon": 4, "const": 4, "petit": 4, "dev": 4, "beta_2": 4, "beta_": 4, "Ici": [4, 5, 6, 7], "r\u00e9duit": 4, "sub": 4, "rappelon": [4, 6, 7], "el": [4, 6], "o": [4, 5], "ignoron": 4, "simplifi": [4, 7], "effectu": [4, 5], "d\u00e9riv": [4, 7], "cha\u00een": [4, 7], "exprim": [4, 6], "partial": [4, 5, 7], "purpl": 4, "red": 4, "sais": 4, "faut": [4, 6, 7], "\u00e9loign": 4, "h\u00e9ritent": 4, "deviennent": 4, "risqu": [4, 6], "\u00e9lev": [4, 5], "tombent": 4, "\u00e9vanescent": [4, 7], "vanishing": [4, 7], "ph\u00e9nomen": [4, 7], "profond": [4, 5], "nombreux": [4, 5, 7], "deuxiem": 4, "r\u00e9pet": [4, 5], "endroit": 4, "d\u00e9velopp": 4, "voyon": 4, "quoi": [4, 7], "ressemblent": [4, 7], "tf": [4, 7], "variabl": [4, 6, 7], "gradienttap": [4, 7], "tape_grad": 4, "tan_x": [4, 7], "tape_sig": 4, "sig_x": 4, "tape_relu": 4, "relu_x": 4, "grad_tanh_x": [4, 7], "grad_sig_x": 4, "grad_relu_x": 4, "lesquel": [4, 5], "null": 4, "concurrent": 4, "appara\u00eet": 4, "optimiseur": 4, "transmis": 4, "moment": 4, "compil": [4, 6], "categorical_crossentropy": [4, 6], "erreur": [4, 5], "quadrat": [4, 5], "binary_crossentropy": 4, "avoir": [4, 7], "contr\u00f4l": [4, 5], "syntax": 4, "optimizer": 4, "very": 4, "good": 4, "ide": 4, "tun": 4, "parameter": 4, "adam_opt": 4, "learning_rat": 4, "001": 4, "order": 4, "use": 4, "custom": 4, "sgd_opt": 4, "phas": 4, "d\u00e9roul": 4, "\u00e9chel": 4, "compar": 4, "similair": [4, 6, 7], "laiss": [4, 7], "to_categorical": [4, 6], "sampl": [4, 6], "drop": [4, 6], "set_random_seed": [4, 6], "256": [4, 6], "metric": [4, 6], "accuracy": [4, 6], "fit": [4, 6], "batch_siz": [4, 6], "30": [4, 6], "verbos": [4, 6], "standardison": 4, "comparon": 4, "obtenu": [4, 6], "std": [4, 6], "h_standardized": 4, "history": [4, 6], "standardis": 4, "Avec": 4, "co\u00fbt": 4, "goh17": 4, "gabriel": 4, "why": 4, "really": 4, "work": 4, "distill": [4, 7], "url": [4, 6, 7], "http": [4, 6, 7], "pub": [4, 7], "kb15": 4, "diederik": 4, "jimmy": 4, "method": 4, "editor": 4, "iclr": 4, "introduir": 5, "cl\u00e9": [5, 7], "d\u00e9taill": 5, "tard": 5, "seaborn": 5, "sn": 5, "terminolog": [5, 7], "uniqu": 5, "underbrac": 5, "reviendron": 5, "compt": 5, "tenu": 5, "observ": [5, 6], "recherch": 5, "usag": 5, "immobili": 5, "essai": 5, "m\u00e9dian": 5, "maison": 5, "occup": 5, "propri\u00e9tair": 5, "milli": 5, "dollar": 5, "piec": 5, "scatterplot": 5, "ayon": 5, "na\u00efv": 5, "laquel": [5, 6], "cherchon": 5, "\u00e8me": 5, "examinon": 5, "to_numpy": 5, "sembl": [5, 6], "autour": 5, "beaucoup": 5, "chos": 5, "acces": 5, "candidat": [5, 6, 7], "fa\u00e7on": [5, 6, 7], "var": 5, "pourrion": 5, "d\u00e9plac": 5, "raid": 5, "initial": 5, "nouveau": 5, "w_": 5, "w_t": 5, "oppos": 5, "vaut": [5, 7], "vectoriel": 5, "processus": [5, 6], "jusqu": 5, "convergent": 5, "1e": 5, "grad_loss": 5, "ww": 5, "w_updat": 5, "Qu": 5, "obtiendr": 5, "prendr": 5, "temp": [5, 7], "Mais": 5, "attent": [5, 7], "toujour": [5, 7], "5e": 5, "voi": 5, "divergeon": 5, "lent": 5, "trop": [5, 6], "briqu": 5, "avanc": 5, "r\u00e9current": 5, "accompagn": 5, "cec": [5, 6], "\u00e9tendu": 5, "forc": 6, "approxim": 6, "universel": 6, "surajust": 6, "overfitting": 6, "formel": 6, "_t": [6, 7], "distribu": 6, "inconnu": 6, "_e": 6, "v\u00e9rit": 6, "mathbb": [6, 7], "sim": 6, "minimiseur": 6, "\u00e9vit": 6, "\u00e9cueil": 6, "techniqu": 6, "conduir": 6, "r\u00e9el": 6, "gr\u00e2c": 6, "myst_nb": 6, "glu": 6, "adam": 6, "validation_spl": 6, "val_loss": 6, "axhlin": 6, "linestyl": 6, "dashed": 6, "102": 6, "epoch_best_model": 6, "argmin": 6, "meilleur": 6, "capac": [6, 7], "g\u00e9n\u00e9ralis": 6, "29": 6, "arr\u00eat": 6, "aurion": 6, "70": 6, "cess": 6, "am\u00e9lior": 6, "tendanc": [6, 7], "oscill": 6, "attend": 6, "souvent": 6, "suppos": 6, "peu": 6, "chanc": 6, "futur": 6, "patienc": 6, "anticip": 6, "configur": 6, "vi": 6, "callback": 6, "earlystopping": 6, "cb_e": 6, "monitor": 6, "restore_best_weight": 6, "epoch_best_model_": 6, "Et": 6, "pr\u00e9vu": 6, "atteint": 6, "cons\u00e9cut": 6, "restaur": 6, "l2": 6, "_r": 6, "_2": 6, "matric": [6, 7], "regularizer": 6, "\u03bb": 6, "kernel_regulariz": 6, "m\u00e9can": 6, "proport": 6, "hasard": 6, "d\u00e9sactiv": 6, "\u00e9tap": 6, "cf": 6, "pr\u00e9senton": 6, "srivastav": 6, "2014": [6, 7], "\u00e9teindr": 6, "changent": 6, "esprit": 6, "for\u00eat": 6, "s\u00e9lection": 6, "divis": 6, "arbre": 6, "int\u00e9rieur": 6, "switchoff_prob": 6, "pourquoi": [6, 7], "presqu": 6, "syst\u00e9mat": 6, "inf\u00e9rieur": 6, "shk": 6, "nitish": 6, "hinton": 6, "alex": 6, "krizhevsky": 6, "ilya": 6, "sutskev": 6, "ruslan": 6, "salakhutdinov": 6, "way": 6, "prevent": 6, "journal": 6, "machin": [6, 7], "research": 6, "56": 6, "1929": 6, "1958": 6, "jmlr": 6, "org": 6, "paper": 6, "v15": 6, "srivastava14": 6, "rnn": 7, "traitent": 7, "\u00e9l\u00e9ment": 7, "temporel": 7, "instant": 7, "x_t": 7, "\u00e9tat": 7, "h_": 7, "proven": 7, "x_0": 7, "diff": 7, "h_t": 7, "w_h": 7, "w_x": 7, "actuel": 7, "index": 7, "partag": 7, "\u00e9chou": 7, "captur": 7, "d\u00e9pend": 7, "mieux": 7, "comprendr": 7, "rappel": 7, "descent": 7, "stochast": 7, "notat": 7, "regardon": 7, "o_t": 7, "cdot": 7, "final": 7, "obtient": 7, "eqnarray": 7, "o_": 7, "influenc": 7, "att\u00e9nu": 7, "facteur": 7, "tap": 7, "point": 7, "rapprochent": 7, "rapid": 7, "absolu": 7, "fer": 7, "tendr": 7, "uns": 7, "pr\u00e9d\u00e9cesseur": 7, "ignor": 7, "actualis": 7, "occurrent": 7, "nom": 7, "lstm": 7, "hochreit": 7, "schmidhub": 7, "1997": 7, "classiqu": 7, "Ils": 7, "visent": 7, "codent": 7, "explicit": 7, "resp": 7, "g": 7, "entrant": 7, "odot": 7, "supprim": 7, "correspondent": 7, "cellul": 7, "c_t": 7, "Cet": 7, "f_t": 7, "i_t": 7, "tild": 7, "forget": 7, "gat": 7, "pouss": 7, "oubli": 7, "inutil": 7, "tour": 7, "partiel": 7, "censur": 7, "laisson": 7, "concern": 7, "concentron": 7, "signific": 7, "apprend": 7, "interm\u00e9diair": 7, "r\u00e9cuper": 7, "propag": 7, "rebour": 7, "dispara\u00eetr": 7, "lien": 7, "encor": 7, "sigm": 7, "w_f": 7, "b_f": 7, "w_i": 7, "b_i": 7, "w_o": 7, "b_o": 7, "concaten": 7, "w_c": 7, "litt\u00e9ratur": 7, "princip": 7, "param\u00e9tris": 7, "l\u00e9ger": 7, "gru": 7, "cho": 7, "grus": 7, "circul": 7, "recour": 7, "z_t": 7, "\u00e9quilibr": 7, "conserv": 7, "r_t": 7, "\u00e9tud": 7, "madsen": 7, "revu": 7, "s\u00e9quentiel": 7, "contraint": 7, "extrair": 7, "discrimin": 7, "exploitent": 7, "derni": 7, "fac": 7, "n\u00e9cessit": 7, "homologu": 7, "cvmerrienboerbb14": 7, "kyunghyun": 7, "bart": 7, "van": 7, "merr": 7, "\u00eb": 7, "nbo": 7, "dzmitry": 7, "bahdanau": 7, "propert": 7, "encod": 7, "decod": 7, "approach": 7, "arxiv": 7, "1409": 7, "1259": 7, "hs97": 7, "sepp": 7, "J": 7, "\u00fc": 7, "rgen": 7, "comput": 7, "1735": 7, "1780": 7, "mad19": 7, "andre": 7, "visualizing": 7, "memoriz": 7}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"r\u00e9seau": [0, 7], "neuronal": [0, 7], "convolut": 0, "neuron": 0, "ser": 0, "temporel": 0, "imag": 0, "convolu": 0, "typ": 0, "lenet": 0, "couch": [0, 3], "padding": 0, "pooling": 0, "ajout": 0, "t\u00eat": 0, "classif": 0, "r\u00e9f\u00e9rent": [0, 6, 7], "introduct": [1, 5], "deep": 1, "learning": 1, "fonction": [2, 3], "co\u00fbt": 2, "erreur": 2, "quadrat": 2, "moyen": 2, "pert": [2, 6], "logist": 2, "perceptron": [3, 5], "multicouch": 3, "empil": 3, "meilleur": 3, "express": 3, "th\u00e9orem": 3, "approxim": 3, "universel": 3, "d\u00e9cid": 3, "architectur": 3, "mlp": 3, "activ": 3, "Le": 3, "cas": 3, "particuli": 3, "sort": 3, "d\u00e9clar": 3, "ker": [3, 4], "exercic": [3, 6], "1": [3, 6], "solut": [3, 6], "2": 3, "3": 3, "optimis": [4, 5], "descent": [4, 5], "gradient": [4, 5], "stochast": 4, "Une": [4, 5], "not": [4, 5], "adam": 4, "La": 4, "mal\u00e9dict": 4, "profondeur": 4, "cod": 4, "tout": 4, "cel": 4, "pr\u00e9trait": 4, "don": 4, "referent": 4, "Un": 5, "premi": 5, "model": 5, "court": 5, "r\u00e9capitul": 5, "r\u00e9gularis": 6, "early": 6, "stopping": 6, "p\u00e9nalis": 6, "dropout": 6, "r\u00e9current": 7, "standard": 7, "long": 7, "short": 7, "term": 7, "memory": 7, "Les": 7, "port": 7, "dan": 7, "gated": 7, "recurrent": 7, "unit": 7, "conclus": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})