Search.setIndex({"docnames": ["content/en/convnets", "content/en/intro", "content/en/loss", "content/en/mlp", "content/en/optim", "content/en/perceptron", "content/en/regularization", "content/en/rnn"], "filenames": ["content/en/convnets.md", "content/en/intro.md", "content/en/loss.md", "content/en/mlp.md", "content/en/optim.md", "content/en/perceptron.md", "content/en/regularization.md", "content/en/rnn.md"], "titles": ["Convolutional Neural Networks", "Deep Learning Basics", "Losses", "Multi Layer Perceptrons", "Optimization", "Introduction", "Regularization", "Recurrent Neural Networks"], "terms": {"aka": 0, "ar": [0, 2, 3, 4, 5, 6, 7], "design": [0, 3, 7], "take": [0, 3, 5, 7], "advantag": 0, "structur": [0, 3], "data": [0, 1, 2, 3, 5, 6, 7], "In": [0, 2, 3, 4, 5, 6, 7], "thi": [0, 1, 2, 3, 4, 6, 7], "chapter": [0, 3, 4, 5, 6, 7], "we": [0, 2, 3, 4, 5, 6, 7], "discuss": [0, 3, 5, 6], "two": [0, 2, 4], "flavour": 0, "start": [0, 4, 5], "monodimension": 0, "case": [0, 2, 4, 5, 7], "see": [0, 4, 5, 6, 7], "how": [0, 3, 4, 5, 7], "1d": 0, "can": [0, 3, 4, 5, 6, 7], "help": [0, 6], "process": [0, 5, 6, 7], "introduc": [0, 3, 5, 6, 7], "2d": 0, "especi": 0, "us": [0, 1, 2, 3, 4, 5, 6, 7], "reli": [0, 3, 4, 7], "oper": 0, "given": [0, 2, 3, 4, 5, 6], "mathbf": [0, 4, 5], "x": [0, 2, 3, 4, 5, 6, 7], "filter": [0, 7], "f": [0, 4], "comput": [0, 3, 4, 5, 6, 7], "an": [0, 4, 5, 6, 7], "activ": [0, 4, 5, 6, 7], "map": [0, 3], "begin": [0, 2, 3, 4, 7], "equat": [0, 3, 7], "left": [0, 3, 4, 5, 6, 7], "right": [0, 3, 4, 5, 6, 7], "t": [0, 3, 4, 5, 7], "sum_": [0, 3, 4, 6, 7], "k": [0, 6], "l": [0, 2, 3, 4, 5, 6], "f_": 0, "x_": [0, 3, 7], "label": [0, 3, 4, 6, 7], "eq": [0, 3], "conv1d": 0, "end": [0, 2, 3, 4, 7], "where": [0, 2, 3, 4, 5, 6, 7], "length": [0, 3], "2l": 0, "1": [0, 2, 4, 5, 7], "The": [0, 1, 2, 5, 6, 7], "follow": [0, 2, 3, 4, 5, 6, 7], "code": [0, 3], "illustr": [0, 4, 5, 6], "notion": [0, 1, 7], "gaussian": 0, "config": [0, 2, 3, 4, 5, 6, 7], "inlinebackend": [0, 2, 3, 4, 5, 6, 7], "figure_format": [0, 2, 3, 4, 5, 6, 7], "svg": [0, 2, 3, 4, 5, 6, 7], "matplotlib": [0, 2, 3, 4, 5, 6, 7], "inlin": [0, 2, 3, 4, 5, 6, 7], "import": [0, 2, 3, 4, 5, 6, 7], "pyplot": [0, 2, 3, 4, 5, 6, 7], "plt": [0, 2, 3, 4, 5, 6, 7], "from": [0, 2, 3, 4, 5, 6, 7], "notebook_util": [0, 2, 3, 4, 5, 6, 7], "prepare_notebook_graph": [0, 2, 3, 4, 5, 6, 7], "numpi": [0, 2, 3, 4, 5, 6, 7], "np": [0, 2, 3, 4, 5, 6], "def": [0, 3, 4, 5, 7], "random_walk": 0, "size": [0, 4, 5], "rnd": 0, "random": [0, 4, 6], "randn": [0, 4], "ts": 0, "rang": [0, 3, 4, 5], "return": [0, 3, 4, 5, 7], "seed": [0, 4], "0": [0, 2, 3, 4, 5, 6, 7], "50": [0, 2, 3, 7], "exp": [0, 3, 4, 7], "linspac": [0, 2, 3, 4, 5, 7], "2": [0, 1, 2, 4, 5, 6, 7], "num": [0, 5], "5": [0, 3, 4, 5], "sum": [0, 3, 4, 5], "figur": [0, 3, 4, 6, 7], "plot": [0, 2, 3, 4, 5, 6, 7], "raw": 0, "correl": 0, "mode": 0, "same": [0, 3, 4, 6, 7], "smooth": [0, 4], "legend": [0, 4, 6, 7], "made": [0, 2, 3, 4, 5], "block": [0, 5, 7], "whose": 0, "paramet": [0, 2, 3, 4, 5, 6, 7], "coeffici": 0, "thei": [0, 3, 4, 6, 7], "emb": 0, "henc": [0, 1, 3, 7], "fix": [0, 3], "priori": [0, 5], "exampl": [0, 3, 4, 6], "abov": [0, 3, 6, 7], "rather": [0, 3, 7], "learn": [0, 3, 4, 5, 6, 7], "These": [0, 4], "translat": [0, 7], "equivari": 0, "which": [0, 2, 3, 4, 5, 6, 7], "mean": [0, 4, 5, 6, 7], "tempor": 0, "shift": 0, "input": [0, 3, 4, 5, 6, 7], "result": [0, 4, 7], "ipython": [0, 4], "displai": [0, 4, 6], "html": [0, 4, 6], "celluloid": 0, "camera": 0, "zero": [0, 4], "12": [0, 3, 4], "4": [0, 2, 3, 4, 5, 6, 7], "8": [0, 3, 4, 7], "60": [0, 3], "fig": [0, 4], "po": 0, "list": 0, "35": 0, "100": [0, 3, 4, 5, 6], "sin": 0, "pi": 0, "act": [0, 6], "subplot": [0, 3, 4], "b": [0, 3, 4, 5, 7], "titl": [0, 3, 4], "fig2": 0, "r": [0, 4, 5, 6, 7], "axes2": 0, "add_ax": 0, "15": [0, 6], "renvoi": 0, "un": 0, "objet": 0, "ax": [0, 4], "set_xtick": 0, "set_titl": [0, 4], "tight_layout": [0, 4], "snap": 0, "anim": [0, 4], "close": [0, 3, 4, 7], "to_jshtml": [0, 4], "tmp": 0, "ipykernel_5308": 0, "368849627": 0, "py": 0, "32": 0, "userwarn": 0, "includ": [0, 1], "compat": 0, "so": [0, 1, 2, 4, 5, 6, 7], "might": [0, 3], "incorrect": 0, "23": [0, 3, 5], "matplotlibdeprecationwarn": 0, "auto": 0, "remov": [0, 7], "overlap": 0, "deprec": 0, "sinc": [0, 3, 4], "3": [0, 4, 5, 6, 7], "6": [0, 3, 4, 5], "minor": 0, "releas": 0, "later": [0, 5], "explicitli": [0, 7], "call": [0, 3, 4, 5, 6, 7], "need": [0, 2, 4], "layout": 0, "ha": [0, 3, 4, 5, 7], "chang": [0, 6], "tight": 0, "onc": [0, 2, 3, 4], "loop": [0, 4], "reflect": [0, 4], "model": [0, 2, 3, 4, 6, 7], "known": [0, 3, 4, 6, 7], "perform": [0, 4, 6], "veri": [0, 3, 4, 5, 6, 7], "well": [0, 3, 4], "vision": 0, "applic": 0, "moder": 0, "amount": [0, 7], "compar": [0, 4], "ones": [0, 3, 6], "cours": [0, 1, 5, 7], "counter": 0, "exist": [0, 3, 7], "term": [0, 3, 4, 5, 6], "vagu": 0, "most": [0, 2, 3], "standard": [0, 2, 4], "architectur": [0, 5, 7], "straight": 0, "forward": 0, "adapt": [0, 4, 7], "commun": 0, "le": [0, 4], "guennec": 0, "et": [0, 6, 7], "al": [0, 6, 7], "2016": 0, "old": 0, "fashion": 0, "altern": [0, 4, 7], "between": [0, 3, 4, 7], "while": [0, 6], "more": [0, 3, 4, 5, 6, 7], "recent": 0, "work": [0, 4], "residu": 0, "incept": 0, "modul": [0, 7], "fawaz": 0, "2020": 0, "basic": [0, 4, 5, 7], "detail": [0, 3, 5, 7], "next": 0, "section": [0, 2, 3, 4, 5, 6], "classif": [0, 1, 2, 3, 4], "present": [0, 2, 3, 4, 5, 6, 7], "benchmark": 0, "2019": [0, 7], "advis": 0, "interest": [0, 3], "reader": 0, "now": [0, 2, 3, 4, 6, 7], "turn": [0, 7], "our": [0, 2, 3, 4, 5], "focu": [0, 7], "slide": 0, "singl": [0, 3, 5], "axi": [0, 4, 5, 6], "dimens": 0, "width": [0, 3], "height": 0, "As": [0, 3, 4, 5, 6], "seen": [0, 3, 4, 6], "below": [0, 3, 4, 6], "pixel": 0, "grid": [0, 2, 3, 4, 7], "each": [0, 3, 4, 5, 6], "intens": 0, "valu": [0, 3, 4, 5, 6, 7], "channel": 0, "color": [0, 3, 4, 6], "typic": [0, 4, 6, 7], "red": [0, 4], "green": 0, "blue": [0, 3, 4], "here": [0, 3, 4, 5, 6, 7], "imread": 0, "cat": 0, "jpg": 0, "image_r": 0, "copi": [0, 3], "image_g": 0, "image_b": 0, "figsiz": [0, 3, 4], "20": [0, 3, 4, 5], "imshow": 0, "rgb": [0, 3], "off": [0, 6], "i": [0, 2, 3, 4, 5], "img": 0, "enumer": 0, "zip": [0, 4], "its": [0, 2, 3, 4, 5, 7], "new": [0, 5, 7], "j": [0, 3, 4, 7], "c": [0, 7], "conv2d": 0, "other": [0, 3, 4, 6, 7], "word": [0, 3, 4, 6, 7], "dot": [0, 4, 7], "product": [0, 7], "tensor": 0, "shape": [0, 3, 4, 7], "2k": 0, "patch": 0, "center": [0, 4], "posit": [0, 3, 4], "let": [0, 3, 4, 5, 7], "consid": [0, 4, 7], "9x9": 0, "sz": 0, "9": [0, 3, 4, 5, 7], "arang": [0, 4, 6], "reshap": [0, 4], "filter_3d": 0, "Then": [0, 7], "greyscal": 0, "ie": 0, "scipi": [0, 4], "signal": 0, "convolve2d": 0, "convoluted_sign": 0, "boundari": 0, "symm": 0, "cmap": 0, "grai": [0, 6], "One": [0, 4, 7], "notic": [0, 3, 4], "blur": 0, "version": [0, 7], "origin": 0, "becaus": [0, 3, 5, 7], "when": [0, 3, 4, 5, 6, 7], "content": 0, "learnt": [0, 5], "than": [0, 3, 4, 6, 7], "set": [0, 2, 3, 4, 6], "lecun": [0, 4], "1998": 0, "stack": 0, "task": [0, 5], "specif": [0, 3, 5], "digit": 0, "recognit": 0, "depict": 0, "A": [0, 6, 7], "sever": [0, 3, 4, 6], "also": [0, 4, 5, 6, 7], "kernel": 0, "parallel": 0, "gener": [0, 3, 4, 5, 6, 7], "all": [0, 2, 3, 4, 6, 7], "form": [0, 3, 4, 7], "share": [0, 3, 7], "bia": [0, 3, 4, 5, 7], "function": [0, 2, 4, 5, 6, 7], "varphi": [0, 3, 4, 5], "prime": [0, 3, 4, 7], "c_": [0, 7], "b_c": [0, 7], "conv_lay": 0, "denot": [0, 2, 3, 6], "note": [0, 1, 3, 6, 7], "associ": [0, 3, 5, 7], "kera": [0, 1, 6], "implement": [0, 6], "class": [0, 2, 3, 4], "keras_cor": [0, 3, 4, 6], "kernel_s": 0, "valid": [0, 6], "relu": [0, 3, 4, 6], "visual": [0, 4, 5, 6, 7], "explan": 0, "sourc": 0, "v": [0, 7], "dumoulin": 0, "visin": 0, "guid": 0, "arithmet": 0, "deep": [0, 4, 5], "without": [0, 4, 5, 6], "With": [0, 4], "ensur": 0, "featur": [0, 3, 4, 7], "achiev": [0, 3], "surround": 0, "area": 0, "repres": [0, 3], "white": 0, "subsampl": 0, "somehow": 0, "summar": [0, 7], "inform": [0, 3, 4, 5, 7], "contain": 0, "lower": [0, 4, 6], "resolut": 0, "idea": [0, 3, 4, 5, 6], "aggreg": 0, "averag": [0, 5, 6], "correspond": [0, 3, 7], "maximum": 0, "max": [0, 4], "order": [0, 2, 3, 4, 6, 7], "reduc": [0, 4], "window": 0, "do": [0, 3, 4, 6, 7], "2x2": 0, "Such": [0, 3], "were": [0, 3], "wide": [0, 2, 3], "earli": 0, "year": 0, "less": 0, "avail": [0, 4], "power": 0, "grow": 0, "through": [0, 3, 6, 7], "maxpool2d": 0, "avgpool2d": 0, "max_pooling_lay": 0, "pool_siz": 0, "average_pooling_lay": 0, "addit": 0, "target": [0, 3, 4, 5, 6], "goal": [0, 3, 5], "probabl": [0, 2, 3], "usual": 0, "head": 0, "consist": [0, 2, 3, 4, 6], "abl": 0, "transform": [0, 7], "vector": [0, 5, 7], "flatten": 0, "sequenti": [0, 3, 4, 6], "inputlay": [0, 3, 4, 6], "dens": [0, 3, 4, 6], "input_shap": [0, 3, 4, 6], "16": [0, 3, 6], "120": [0, 3, 5], "84": 0, "10": [0, 3, 4, 5, 6], "softmax": [0, 3, 4, 6], "summari": [0, 3, 4], "_________________________________________________________________": [0, 3, 4], "type": [0, 3, 4], "param": [0, 3, 4], "none": [0, 3, 4, 5], "28": 0, "156": 0, "max_pooling2d": 0, "maxpooling2": 0, "14": [0, 6], "d": [0, 2, 3, 4, 6, 7], "conv2d_1": 0, "2416": 0, "max_pooling2d_1": 0, "maxpoolin": 0, "g2d": 0, "400": 0, "48120": 0, "dense_1": [0, 3, 4], "10164": 0, "dense_2": 0, "850": 0, "total": [0, 3, 4], "61706": 0, "241": 0, "04": 0, "kb": [0, 3, 4], "trainabl": [0, 3, 4], "non": [0, 3, 4], "00": [0, 3, 4], "byte": [0, 3, 4], "ffw": 0, "19": 0, "hassan": 0, "ismail": 0, "germain": 0, "foresti": 0, "jonathan": 0, "weber": 0, "lhassan": 0, "idoumghar": 0, "pierr": 0, "alain": 0, "muller": 0, "review": [0, 7], "mine": 0, "knowledg": 0, "discoveri": 0, "33": [0, 3, 5], "917": 0, "963": 0, "flf": 0, "benjamin": 0, "luca": 0, "charlott": 0, "pelleti": 0, "daniel": 0, "schmidt": 0, "geoffrei": [0, 6], "webb": 0, "fran": 0, "\u00e7": 0, "oi": 0, "petitjean": 0, "inceptiontim": 0, "find": [0, 5], "alexnet": 0, "34": [0, 3, 5], "1936": 0, "1962": 0, "lgmt16": 0, "arthur": 0, "simon": 0, "malinowski": 0, "romain": [0, 1], "tavenard": [0, 1], "augment": 0, "ecml": 0, "pkdd": 0, "workshop": 0, "advanc": [0, 5], "analyt": 0, "riva": 0, "del": 0, "garda": 0, "itali": 0, "septemb": 0, "lbbh98": 0, "yann": [0, 4], "\u00e9": 0, "bottou": 0, "yoshua": [0, 4, 7], "bengio": [0, 4, 7], "patrick": 0, "haffner": 0, "gradient": [0, 7], "base": [0, 3, 4, 7], "appli": [0, 3, 4, 7], "document": [0, 1], "proceed": 0, "ieee": 0, "86": 0, "11": [0, 3, 4, 5], "2278": 0, "2324": 0, "serv": 1, "lectur": 1, "taught": 1, "universit\u00e9": 1, "de": 1, "renn": 1, "franc": 1, "edhec": 1, "lill": 1, "deal": [1, 4, 5], "neural": [1, 2, 3, 4, 5, 6], "network": [1, 2, 3, 4, 5, 6], "regress": [1, 2, 3, 5], "over": [1, 4, 5, 6, 7], "tabular": 1, "optim": [1, 2, 3, 6, 7], "algorithm": [1, 3, 4, 5, 7], "multi": [1, 4, 5], "layer": [1, 4, 5, 6], "perceptron": [1, 4], "convolut": [1, 5, 7], "imag": 1, "transfer": 1, "sequenc": [1, 7], "forecast": 1, "lab": 1, "have": [2, 3, 4, 5, 6, 7], "first": [2, 3, 4, 7], "famili": [2, 3, 5], "mlp": 2, "train": [2, 4, 6, 7], "e": [2, 3, 4, 5, 6], "tune": [2, 4], "fit": [2, 4, 6], "defin": [2, 3, 4, 7], "inde": [2, 3, 4, 7], "pick": [2, 3, 5, 6], "minim": [2, 4, 5, 6], "mainli": 2, "assum": [2, 5, 6], "dataset": [2, 3, 4, 5, 6, 7], "mathcal": [2, 4, 5, 6], "n": [2, 4], "annot": 2, "sampl": [2, 4, 5, 6], "x_i": [2, 4, 5, 6], "y_i": [2, 4, 5, 6], "s": [2, 3, 4, 5], "output": [2, 4, 5, 7], "foral": [2, 3, 7], "hat": [2, 3, 4, 5], "y": [2, 3, 4, 5, 6], "_i": [2, 3, 4], "m_": [2, 4, 6], "theta": [2, 4, 6], "weight": [2, 3, 4, 5, 6, 7], "bias": [2, 3, 4], "mse": [2, 4, 5], "commonli": 2, "It": [2, 3, 4, 5], "align": [2, 3, 4], "frac": [2, 3, 4, 5, 6, 7], "sum_i": [2, 3, 4, 5], "Its": 2, "quadrat": 2, "formul": [2, 3, 7], "tend": [2, 6, 7], "strongli": 2, "penal": 2, "larg": [2, 4, 5, 6], "xlabel": [2, 4, 6], "ylabel": [2, 4, 6], "log": [2, 4], "p": [2, 4], "predict": [2, 3, 4, 5, 6], "correct": 2, "favor": 2, "expect": [2, 3], "ion": 2, "01": [2, 6], "previou": [3, 4, 5, 6, 7], "simpl": [3, 5, 6], "linear": [3, 5], "combin": 3, "plu": 3, "x_j": 3, "w_j": 3, "among": 3, "quit": 3, "restrict": 3, "cover": 3, "wider": [3, 4], "one": [3, 4, 5, 6, 7], "neuron": [3, 5, 6], "organ": 3, "complex": 3, "hidden": [3, 4, 6, 7], "extra": [3, 4, 7], "question": 3, "ask": 3, "whether": [3, 4], "ad": 3, "effect": [3, 4, 7], "allow": [3, 7], "what": [3, 4, 5, 7], "about": [3, 4, 5, 7], "state": [3, 7], "ani": [3, 4, 6], "continu": [3, 6], "compact": 3, "want": 3, "sigmoid": [3, 4, 7], "properti": [3, 7], "howev": [3, 6], "number": [3, 5, 6], "necessari": 3, "qualiti": 3, "moreov": 3, "suffici": [3, 6], "good": [3, 4, 5], "anoth": [3, 4, 6, 7], "eventu": 3, "converg": [3, 5], "guarante": 3, "dedic": [3, 5], "practic": [3, 4], "observ": [3, 5, 6], "empir": [3, 6], "effici": 3, "requir": [3, 4, 7], "graphic": [3, 7], "represent": [3, 7], "varphi_": 3, "text": [3, 4, 5, 7], "out": [3, 4, 7], "w": [3, 4, 5, 7], "_": [3, 4, 5, 6], "teal": 3, "h": [3, 4, 6, 7], "sum_j": 3, "ij": 3, "61": 3, "91": 3, "mlp_2hidden": 3, "To": [3, 4, 5, 6, 7], "even": [3, 6], "precis": 3, "problem": [3, 4, 5], "some": [3, 4, 5, 6], "quantiti": [3, 4, 5], "hand": [3, 4, 6], "hyper": [3, 4, 5], "iri": [3, 4, 6], "panda": [3, 4, 5, 6], "pd": [3, 4, 5, 6], "read_csv": [3, 4, 5, 6], "csv": [3, 4, 5, 6], "index_col": [3, 4, 6], "sepal": 3, "cm": 3, "petal": 3, "7": [3, 4, 5], "145": 3, "146": 3, "147": [3, 5], "148": 3, "149": 3, "150": [3, 4], "row": [3, 5], "column": [3, 4, 5, 6], "infer": 3, "attribut": 3, "differ": [3, 4, 6, 7], "possibl": 3, "dictat": 3, "equal": 3, "descript": 3, "per": [3, 4, 5, 7], "face": [3, 7], "situat": 3, "stake": [3, 5], "come": [3, 5], "binari": [3, 4], "indic": 3, "mani": [3, 4, 7], "choic": 3, "ident": [3, 5], "whatev": 3, "depth": 3, "would": [3, 5, 6, 7], "fall": 3, "back": [3, 5], "onli": [3, 4, 5, 6, 7], "regim": 3, "don": 3, "behav": [3, 4], "like": [3, 4, 7], "whole": [3, 4, 6], "histor": 3, "been": [3, 7], "propos": 3, "tanh": [3, 4, 7], "2x": 3, "gt": 3, "otherwis": 3, "ylim": [3, 4], "variant": [3, 4, 7], "nowadai": 3, "reason": 3, "you": [3, 5, 6], "provid": [3, 7], "own": 3, "bit": 3, "adequ": 3, "suppos": [3, 5], "If": [3, 6], "wa": [3, 6], "boston": [3, 5], "hous": [3, 5], "price": [3, 5], "nonneg": 3, "earlier": 3, "lie": 3, "interv": [3, 4], "default": 3, "final": [3, 7], "context": 3, "should": [3, 4, 5, 6, 7], "For": 3, "purpos": 3, "o_i": 3, "o_j": 3, "befor": [3, 4, 6], "just": 3, "look": [3, 4, 5, 7], "unit": [3, 4, 6], "tensorflow": [3, 4, 6, 7], "backend": [3, 4, 6], "220": [3, 4], "63": [3, 4], "283": [3, 4], "overview": 3, "explain": [3, 6], "fulli": 3, "connect": 3, "alreadi": 3, "make": [3, 4, 5, 7], "time": [3, 4, 5, 7], "200": 3, "similarli": [3, 7], "those": [3, 4, 7], "overal": [3, 4, 6], "full": [3, 4, 6, 7], "shown": 3, "rm": [3, 5], "crim": 3, "indu": 3, "nox": 3, "ag": 3, "tax": 3, "575": [3, 5], "00632": 3, "31": 3, "538": 3, "65": 3, "296": 3, "24": [3, 5], "421": [3, 5], "02731": 3, "07": 3, "469": 3, "78": 3, "242": 3, "21": [3, 5], "185": [3, 5], "02729": 3, "998": [3, 5], "03237": 3, "18": 3, "458": 3, "45": 3, "222": 3, "06905": 3, "54": 3, "36": [3, 5], "501": [3, 5], "593": [3, 5], "06263": 3, "93": 3, "573": 3, "69": 3, "273": 3, "22": [3, 5], "502": [3, 5], "04527": 3, "76": 3, "503": [3, 5], "976": [3, 5], "06076": 3, "504": [3, 5], "794": [3, 5], "10959": 3, "89": 3, "505": [3, 5], "030": [3, 5], "04741": 3, "80": 3, "506": [3, 5], "strategi": [4, 5, 6], "show": 4, "limit": [4, 7], "initi": [4, 5], "nabla_": [4, 7], "updat": [4, 7], "rule": [4, 7], "iter": [4, 5], "leftarrow": 4, "rho": [4, 5], "method": [4, 5], "rate": [4, 5, 6], "direct": [4, 5], "steepest": [4, 5], "decreas": 4, "loss": [4, 5], "epoch": [4, 6], "pass": 4, "occur": 4, "strong": 4, "motiv": 4, "behind": [4, 5, 6], "get": [4, 5, 7], "cheap": 4, "estim": [4, 6], "draw": 4, "subset": 4, "minibatch": [4, 6], "interestingli": 4, "after": [4, 6], "multipl": [4, 7], "n_": 4, "level": 4, "consequ": 4, "frequent": 4, "noisi": 4, "instead": [4, 7], "true": [4, 6], "grad": 4, "alpha": [4, 5], "lambd": 4, "norm": 4, "sqrt": 4, "cost": 4, "todo": 4, "pour": 4, "pa": 4, "que": 4, "fass": 4, "nimp": 4, "optim_gd": 4, "alpha_init": 4, "n_epoch": [4, 6], "append": [4, 5], "concaten": [4, 7], "optim_sgd": 4, "minibatch_s": 4, "scaled_lambda": 4, "indices_minibatch": 4, "randint": 4, "x_minibatch": 4, "y_minibatch": 4, "stretch_to_rang": 4, "lim": 4, "sz_rang": 4, "middl": 4, "get_lim": 4, "alphas_list": 4, "xlim": [4, 6], "min": [4, 6], "els": [4, 5], "gen_anim": 4, "alphas_gd": 4, "alphas_sgd": 4, "alpha_star": 4, "n_steps_per_epoch": 4, "gen_video": 4, "global": 4, "lines_alpha": 4, "40": [4, 6], "nn": 4, "xv": 4, "yv": 4, "meshgrid": 4, "xvisu": 4, "ravel": 4, "pv": 4, "13": 4, "contour": 4, "ko": [4, 5], "fillstyl": 4, "line_alpha": 4, "marker": 4, "set_xlabel": 4, "w_0": [4, 5], "set_ylabel": 4, "w_1": 4, "set_xlim": 4, "set_ylim": 4, "text_epoch": 4, "set_xdata": 4, "set_ydata": 4, "set_text": 4, "funcanim": 4, "500": 4, "blit": 4, "fals": [4, 6], "save_count": 4, "len": [4, 6], "rand": 4, "astyp": 4, "int": 4, "2e": 4, "arrai": [4, 5], "res_optim": 4, "fun": 4, "lambda": [4, 6], "x0": 4, "jac": 4, "is_html_output": 4, "viz": 4, "repeat": [4, 5], "apart": 4, "impli": 4, "benefit": 4, "kei": [4, 5, 7], "contrari": 4, "had": [4, 6], "logist": 4, "longer": 4, "convex": 4, "soon": [4, 6], "least": 4, "model_forward_loss": 4, "0001": 4, "w0": 4, "75": 4, "wi": 4, "suffer": 4, "local": [4, 5, 7], "optima": 4, "landscap": 4, "seriou": 4, "gd": 4, "On": [4, 7], "escap": 4, "minima": 4, "kingma": 4, "ba": 4, "2015": 4, "definit": [4, 5], "step": [4, 5, 6], "momentum": 4, "past": [4, 7], "trajectori": 4, "space": 4, "dure": [4, 6], "interact": 4, "found": 4, "goh": 4, "2017": 4, "plugin": 4, "replac": 4, "m": 4, "beta_1": 4, "balanc": [4, 7], "current": [4, 6, 7], "store": [4, 5], "theta_i": 4, "epsilon": 4, "small": 4, "constant": 4, "beta_2": 4, "beta_": 4, "recal": [4, 6, 7], "ell": [4, 6], "o": [4, 5], "ignor": [4, 7], "simplifi": [4, 7], "respect": [4, 5, 7], "By": 4, "chain": [4, 7], "express": [4, 6], "partial": [4, 5, 7], "purpl": 4, "There": [4, 7], "insight": 4, "grasp": 4, "further": [4, 5, 7], "inherit": 4, "smaller": [4, 5], "higher": 4, "risk": [4, 6], "collaps": 4, "vanish": [4, 7], "common": 4, "phenomenon": [4, 7], "second": 4, "formula": [4, 7], "place": 4, "develop": 4, "inspect": 4, "deriv": [4, 7], "tf": [4, 7], "variabl": [4, 6, 7], "gradienttap": [4, 7], "tape_grad": 4, "tan_x": [4, 7], "tape_sig": 4, "sig_x": 4, "tape_relu": 4, "relu_x": 4, "grad_tanh_x": [4, 7], "grad_sig_x": 4, "grad_relu_x": 4, "competitor": 4, "attract": 4, "candid": [4, 5, 6, 7], "appear": 4, "repeatedli": 4, "compil": [4, 6], "categorical_crossentropi": [4, 6], "squar": [4, 5], "error": [4, 5], "binary_crossentropi": 4, "control": [4, 5], "syntax": 4, "Not": 4, "adam_opt": 4, "learning_r": 4, "001": 4, "custom": 4, "sgd_opt": 4, "phase": 4, "scale": [4, 5], "similar": [4, 6, 7], "both": [4, 5, 6, 7], "util": [4, 6], "to_categor": [4, 6], "drop": [4, 6], "set_random_se": [4, 6], "256": [4, 6], "metric": [4, 6], "accuraci": [4, 6], "batch_siz": [4, 6], "30": [4, 6], "verbos": [4, 6], "std": [4, 6], "h_standard": 4, "histori": [4, 6], "goh17": 4, "gabriel": 4, "why": [4, 6, 7], "realli": 4, "distil": [4, 7], "url": [4, 6, 7], "http": [4, 6, 7], "pub": [4, 7], "kb15": 4, "diederik": 4, "jimmi": 4, "editor": 4, "iclr": 4, "wai": [5, 6, 7], "concept": [5, 6], "seaborn": 5, "sn": 5, "terminolog": [5, 7], "parametr": [5, 7], "underbrac": 5, "chosen": 5, "book": 5, "aim": [5, 7], "solv": 5, "enough": 5, "coin": 5, "field": 5, "extens": 5, "mind": 5, "try": 5, "median": 5, "owner": 5, "occupi": 5, "home": 5, "1000": 5, "room": 5, "dwell": 5, "scatterplot": 5, "forc": 5, "naiv": 5, "approach": [5, 7], "intercept": 5, "object": [5, 6], "ground": 5, "truth": 5, "th": 5, "to_numpi": 5, "seem": [5, 6], "around": 5, "lot": 5, "cannot": 5, "someth": 5, "access": 5, "vari": 5, "could": 5, "move": 5, "w_": 5, "w_t": 5, "done": 5, "evalu": [5, 6], "opposit": 5, "point": 5, "hold": 5, "too": [5, 6], "until": 5, "1e": 5, "grad_loss": 5, "ww": 5, "w_updat": 5, "But": 5, "care": 5, "larger": [5, 7], "alwai": 5, "5e": 5, "slowli": 5, "diverg": 5, "build": 5, "recurr": 5, "fact": [5, 6], "extend": 5, "strength": 6, "approxim": 6, "univers": 6, "machin": [6, 7], "relat": 6, "overfit": 6, "formal": 6, "_t": [6, 7], "drawn": 6, "unknown": 6, "distribut": 6, "_e": 6, "wherea": 6, "real": 6, "mathbb": [6, 7], "sim": 6, "avoid": 6, "pitfal": 6, "techniqu": 6, "lead": 6, "myst_nb": 6, "glue": 6, "adam": 6, "validation_split": 6, "val_loss": 6, "axhlin": 6, "linestyl": 6, "dash": 6, "102": 6, "epoch_best_model": 6, "argmin": 6, "best": 6, "capabl": 6, "gotten": 6, "better": [6, 7], "70": 6, "improv": 6, "oscil": 6, "often": 6, "wait": 6, "unlik": 6, "futur": 6, "patienc": 6, "up": 6, "via": 6, "callback": 6, "earlystop": 6, "cb_e": 6, "monitor": 6, "restore_best_weight": 6, "epoch_best_model_": 6, "And": 6, "schedul": 6, "reach": 6, "consecut": 6, "restor": 6, "enforc": 6, "instanc": 6, "l2": 6, "_r": 6, "_2": 6, "matrix": [6, 7], "shrink": 6, "\u03bb": 6, "kernel_regular": 6, "mechan": 6, "mini": 6, "batch": 6, "proport": 6, "switch": 6, "subsequ": 6, "sub": 6, "cf": 6, "side": 6, "colour": 6, "srivastava": 6, "2014": [6, 7], "spirit": 6, "forest": 6, "randomli": 6, "select": 6, "tree": 6, "split": 6, "insid": 6, "main": 6, "switchoff_proba": 6, "almost": 6, "subpart": 6, "retriev": 6, "measur": 6, "shk": 6, "nitish": 6, "hinton": 6, "alex": 6, "krizhevski": 6, "ilya": 6, "sutskev": 6, "ruslan": 6, "salakhutdinov": 6, "prevent": 6, "journal": 6, "research": 6, "56": 6, "1929": 6, "1958": 6, "jmlr": 6, "org": 6, "paper": 6, "v15": 6, "srivastava14a": 6, "proce": 7, "element": 7, "seri": 7, "x_t": 7, "h_": 7, "x_0": 7, "variou": 7, "mostli": 7, "h_t": 7, "w_h": 7, "w_x": 7, "index": 7, "across": 7, "timestamp": 7, "easili": 7, "fail": 7, "captur": 7, "depend": 7, "understand": 7, "remind": 7, "stochast": 7, "descent": 7, "notat": 7, "scalar": 7, "actual": 7, "o_t": 7, "cdot": 7, "eqnarrai": 7, "o_": 7, "influenc": 7, "mitig": 7, "factor": 7, "tape": 7, "quickli": 7, "absolut": 7, "few": 7, "predecessor": 7, "occurr": 7, "lstm": 7, "hochreit": 7, "schmidhub": 7, "1997": 7, "encod": 7, "piec": 7, "resp": 7, "kept": 7, "g": 7, "incom": 7, "odot": 7, "wise": 7, "part": 7, "low": 7, "cell": 7, "c_t": 7, "f_t": 7, "i_t": 7, "tild": 7, "forget": 7, "push": 7, "useless": 7, "partli": 7, "censor": 7, "delai": 7, "significantli": 7, "recov": 7, "flow": 7, "anymor": 7, "link": 7, "sigma": 7, "w_f": 7, "b_f": 7, "w_i": 7, "b_i": 7, "w_o": 7, "b_o": 7, "w_c": 7, "literatur": 7, "still": 7, "principl": 7, "slightli": 7, "gat": 7, "gru": 7, "cho": 7, "signific": 7, "though": 7, "resort": 7, "z_t": 7, "r_t": 7, "hide": 7, "studi": 7, "abil": 7, "madsen": 7, "constraint": 7, "tackl": 7, "attent": 7, "extract": 7, "discrimin": 7, "leverag": 7, "concern": 7, "latter": 7, "counterpart": 7, "meaning": 7, "cvmerrienboerbb14": 7, "kyunghyun": 7, "bart": 7, "van": 7, "merri": 7, "\u00eb": 7, "nboer": 7, "dzmitri": 7, "bahdanau": 7, "decod": 7, "arxiv": 7, "1409": 7, "1259": 7, "hs97": 7, "sepp": 7, "\u00fc": 7, "rgen": 7, "1735": 7, "1780": 7, "mad19": 7, "andrea": 7, "memor": 7}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"convolut": 0, "neural": [0, 7], "network": [0, 7], "convnet": 0, "time": 0, "seri": 0, "imag": 0, "cnn": 0, "\u00e0": 0, "la": 0, "lenet": 0, "layer": [0, 3], "pad": 0, "pool": 0, "plug": 0, "fulli": 0, "connect": 0, "output": [0, 3], "refer": [0, 4, 6, 7], "deep": 1, "learn": 1, "basic": 1, "loss": [2, 6], "mean": 2, "squar": 2, "error": 2, "logist": 2, "multi": 3, "perceptron": [3, 5], "stack": 3, "better": 3, "express": 3, "univers": 3, "approxim": 3, "theorem": 3, "decid": 3, "an": 3, "mlp": 3, "architectur": 3, "activ": 3, "function": 3, "The": [3, 4], "special": 3, "case": 3, "declar": 3, "kera": [3, 4], "exercis": [3, 6], "1": [3, 6], "solut": [3, 6], "2": 3, "3": 3, "optim": [4, 5], "gradient": [4, 5], "descent": [4, 5], "stochast": 4, "sgd": 4, "A": [4, 5], "note": [4, 5], "adam": 4, "curs": 4, "depth": 4, "wrap": [4, 5], "thing": 4, "up": [4, 5], "data": 4, "preprocess": 4, "introduct": 5, "first": 5, "model": 5, "short": [5, 7], "thi": 5, "regular": 6, "earli": 6, "stop": 6, "penal": 6, "dropout": 6, "recurr": 7, "vanilla": 7, "rnn": 7, "long": 7, "term": 7, "memori": 7, "gate": 7, "unit": 7, "conclus": 7}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})