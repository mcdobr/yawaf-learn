# One class classification prototype for learning
# Tutorials followed:
# One class classification https://machinelearningmastery.com/one-class-classification-algorithms/
# Multiple models and pandas usage https://lukesingham.com/whos-going-to-leave-next/
# Examples of tuning parameters
# https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/
import collections
import csv
import datetime
import re
import urllib.parse

import numpy as np
import pandas as pd
from matplotlib import pyplot
from skl2onnx import to_onnx
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM, SVC, SVR
from sklearn.tree import DecisionTreeClassifier

NORMAL_LABEL = 0
ANOMALOUS_LABEL = 1


def attribute_length(request_dict):
    method = request_dict.get('Method')
    if method == 'GET':
        return len(request_dict.get('Query'))
    else:
        return len(request_dict.get('Body'))


def sql_keywords(request_dict, split_pattern='\\W'):
    """Count the number of SQL keywords"""
    sql2016_keywords = {
        "abs", "acos", "all", "allocate", "alter", "and", "any", "are", "array", "array_agg", "array_max_cardinality",
        "as",
        "asensitive", "asin", "asymmetric", "at", "atan", "atomic", "authorization", "avg", "begin", "begin_frame",
        "begin_partition", "between", "bigint", "binary", "blob", "boolean", "both", "by", "call", "called",
        "cardinality",
        "cascaded", "case", "cast", "ceil", "ceiling", "char", "character", "character_length", "char_length", "check",
        "classifier", "clob", "close", "coalesce", "collate", "collect", "column", "commit", "condition", "connect",
        "constraint", "contains", "convert", "copy", "corr", "corresponding", "cos", "cosh", "count", "covar_pop",
        "covar_samp", "create", "cross", "cube", "cume_dist", "current", "current_catalog", "current_date",
        "current_default_transform_group", "current_path", "current_role", "current_row", "current_schema",
        "current_time",
        "current_timestamp", "current_transform_group_for_type", "current_user", "cursor", "cycle", "date", "day",
        "deallocate", "dec", "decfloat", "decimal", "declare", "default", "define", "delete", "dense_rank", "deref",
        "describe", "deterministic", "disconnect", "distinct", "double", "drop", "dynamic", "each", "element", "else",
        "empty", "end", "end-exec", "end_frame", "end_partition", "equals", "escape", "every", "except", "exec",
        "execute", "exists", "exp", "external", "extract", "false", "fetch", "filter", "first_value", "float", "floor",
        "for", "foreign", "frame_row", "free", "from", "full", "function", "fusion", "get", "global", "grant", "group",
        "grouping", "groups", "having", "hold", "hour", "identity", "in", "indicator", "initial", "inner", "inout",
        "insensitive", "insert", "int", "integer", "intersect", "intersection", "interval", "into", "is", "join",
        "json_array", "json_arrayagg", "json_exists", "json_object", "json_objectagg", "json_query", "json_table",
        "json_table_primitive", "json_value", "lag", "language", "large", "last_value", "lateral", "lead", "leading",
        "left", "like", "like_regex", "listagg", "ln", "local", "localtime", "localtimestamp", "log", "log10", "lower",
        "match", "matches", "match_number", "match_recognize", "max", "member", "merge", "method", "min", "minute",
        "mod",
        "modifies", "module", "month", "multiset", "national", "natural", "nchar", "nclob", "new", "no", "none",
        "normalize", "not", "nth_value", "ntile", "null", "nullif", "numeric", "occurrences_regex", "octet_length",
        "of",
        "offset", "old", "omit", "on", "one", "only", "open", "or", "order", "out", "outer", "over", "overlaps",
        "overlay",
        "parameter", "partition", "pattern", "per", "percent", "percentile_cont", "percentile_disc", "percent_rank",
        "period", "portion", "position", "position_regex", "power", "precedes", "precision", "prepare", "primary",
        "procedure", "ptf", "range", "rank", "reads", "real", "recursive", "ref", "references", "referencing",
        "regr_avgx",
        "regr_avgy", "regr_count", "regr_intercept", "regr_r2", "regr_slope", "regr_sxx", "regr_sxy", "regr_syy",
        "release",
        "result", "return", "returns", "revoke", "right", "rollback", "rollup", "row", "rows", "row_number", "running",
        "savepoint", "scope", "scroll", "search", "second", "seek", "select", "sensitive", "session_user", "set",
        "show",
        "similar", "sin", "sinh", "skip", "smallint", "some", "specific", "specifictype", "sql", "sqlexception",
        "sqlstate",
        "sqlwarning", "sqrt", "start", "static", "stddev_pop", "stddev_samp", "submultiset", "subset", "substring",
        "substring_regex", "succeeds", "sum", "symmetric", "system", "system_time", "system_user", "table",
        "tablesample",
        "tan", "tanh", "then", "time", "timestamp", "timezone_hour", "timezone_minute", "to", "trailing", "translate",
        "translate_regex", "translation", "treat", "trigger", "trim", "trim_array", "true", "truncate", "uescape",
        "union",
        "unique", "unknown", "unnest", "update", "upper", "user", "using", "value", "values", "value_of", "varbinary",
        "varchar", "varying", "var_pop", "var_samp", "versioning", "when", "whenever", "where", "width_bucket",
        "window",
        "with", "within", "without", "year"
    }

    method = request_dict.get('Method')
    if method == 'GET':
        target = request_dict.get('Query').lower()
    else:
        target = request_dict.get('Body').lower()
    words = filter(None, re.split(split_pattern, target))
    appearances = count_appearances(words, sql2016_keywords)
    return sum(appearances.values())


def total_length(request_dict):
    """Counts the parameter names (including maybe custom headers that are tried) and value lengths to get
     close to provide an approximation into the size of the actual raw request."""
    length = 0
    for key, value in request_dict.items():
        if key == 'Class':
            pass
        length += len(key) + len(value)
    return length


def number_of_letters(request_dict):
    result = 0
    for key, value in request_dict.items():
        result += sum(c.isalpha() for c in key) + sum(c.isalpha() for c in value)
    return result


def non_printable_characters(request_dict):
    result = 0
    for key, value in request_dict.items():
        result += sum(not c.isprintable() for c in key) + sum(not c.isprintable() for c in value)
    return result


def entropy(request_dict):
    reconstructed_request = ''.join(list(request_dict.values()))
    return -1 * sum(i / len(reconstructed_request) for i in collections.Counter(reconstructed_request).values())


def url_length(request_dict):
    return len(request_dict['Path']) + len(request_dict['Query'])


def path_non_alpha(request_dict):
    return sum(not c.isalpha() for c in request_dict['Path'])


def count_js_keywords(request_dict, split_pattern='\\W'):
    javascript_keywords = {
        "abstract", "alert", "all", "anchor", "anchors", "area", "arguments", "assign", "await", "blur", "boolean",
        "break",
        "button", "byte", "case", "catch", "charclass", "checkbox", "clearInterval", "clearTimeout",
        "clientInformation",
        "close", "closed", "confirm", "const", "constructor", "continue", "crypto", "debugger", "decodeURI",
        "decodeURIComponent", "default", "defaultStatus", "delete", "do", "document", "double", "element", "elements",
        "else", "embed", "embeds", "encodeURI", "encodeURIComponent", "enum", "escape", "eval", "event", "export",
        "extends", "false", "fileUpload", "final", "finally", "float", "focus", "for", "form", "forms", "frame",
        "frameRate", "frames", "function", "goto", "hidden", "history", "if", "image", "images", "implements", "import",
        "in", "innerHeight", "innerWidth", "instanceof", "int", "interface", "layer", "layers", "let", "link",
        "location",
        "long", "mimeTypes", "native", "navigate", "navigator", "new", "null", "offscreenBuffering", "open", "opener",
        "option", "outerHeight", "outerWidth", "package", "packages", "pageXOffset", "pageYOffset", "parent",
        "parseFloat",
        "parseInt", "password", "pkcs11" "plugin", "private", "prompt", "propertyIsEnum", "protected", "public",
        "radio",
        "reset", "return", "screenX", "screenY", "scroll", "secure", "select", "self", "setInterval", "setTimeout",
        "short",
        "static", "status", "submit", "super", "switch", "synchronized", "taint", "text", "textarea", "this", "throw",
        "throws", "top", "transient", "true", "try", "typeof", "unescape", "untaint", "var", "void", "volatile",
        "while",
        "window", "with", "yield"
    }

    result = 0
    for key, value in request_dict.items():
        words = re.split(split_pattern, value)
        result += sum(count_appearances(words, javascript_keywords).values())
    return result


def count_event_handlers(request_dict, split_pattern='\\W'):
    """Count the number of HTML event handlers (e.g onclick, onmouseover etc.) in a request."""
    event_handlers = {
        "onactivate", "onafterprint", "onafterscriptexecute", "onanimationcancel", "onanimationend",
        "onanimationiteration",
        "onanimationstart", "onauxclick", "onbeforeactivate", "onbeforecopy", "onbeforecut", "onbeforedeactivate",
        "onbeforepaste", "onbeforeprint", "onbeforescriptexecute", "onbeforeunload", "onbegin", "onblur", "onbounce",
        "oncanplay", "oncanplaythrough", "onchange", "onclick", "onclose", "oncontextmenu", "oncopy", "oncuechange",
        "oncut", "ondblclick", "ondeactivate", "ondrag", "ondragend", "ondragenter", "ondragleave", "ondragover",
        "ondragstart", "ondrop", "ondurationchange", "onend", "onended", "onerror", "onfinish", "onfocus", "onfocusin",
        "onfocusout", "onfullscreenchange", "onhashchange", "oninput", "oninvalid", "onkeydown", "onkeypress",
        "onkeyup",
        "onload", "onloadeddata", "onloadedmetadata", "onloadend", "onloadstart", "onmessage", "onmousedown",
        "onmouseenter", "onmouseleave", "onmousemove", "onmouseout", "onmouseover", "onmouseup", "onmousewheel",
        "onmozfullscreenchange", "onpagehide", "onpageshow", "onpaste", "onpause", "onplay", "onplaying",
        "onpointerdown",
        "onpointerenter", "onpointerleave", "onpointermove", "onpointerout", "onpointerover", "onpointerrawupdate",
        "onpointerup", "onpopstate", "onprogress", "onreadystatechange", "onrepeat", "onreset", "onresize", "onscroll",
        "onsearch", "onseeked", "onseeking", "onselect", "onselectionchange", "onselectstart", "onshow", "onstart",
        "onsubmit", "ontimeupdate", "ontoggle", "ontouchend", "ontouchmove", "ontouchstart", "ontransitioncancel",
        "ontransitionend", "ontransitionrun", "ontransitionstart", "onunhandledrejection", "onunload", "onvolumechange",
        "onwaiting", "onwebkitanimationend", "onwebkitanimationiteration", "onwebkitanimationstart",
        "onwebkittransitionend", "onwheel"
    }
    result = 0
    for value in request_dict.values():
        words = re.split(split_pattern, value)
        result += sum(count_appearances(words, event_handlers).values())
    return result


def count_html_tags(request_dict, split_pattern='\\W'):
    html_tags = {
        "a", "abbr", "acronym", "address", "animate", "animatemotion", "animatetransform", "applet", "area",
        "article", "aside", "audio", "b", "base", "basefont", "bdi", "bdo", "bgsound", "big", "blink",
        "blockquote", "body", "br", "button", "canvas", "caption", "center", "cite", "code", "col", "colgroup",
        "command", "content", "custom", "tags", "data", "datalist", "dd", "del", "details", "dfn", "dialog",
        "dir", "div", "dl", "dt", "element", "em", "embed", "fieldset", "figcaption", "figure", "font",
        "footer", "form", "frame", "frameset", "h1", "head", "header", "hgroup", "hr", "html", "i", "iframe",
        "image", "img", "input", "ins", "isindex", "kbd", "keygen", "label", "legend", "li", "link", "listing",
        "main", "map", "mark", "marquee", "menu", "menuitem", "meta", "meter", "multicol", "nav", "nextid",
        "nobr", "noembed", "noframes", "noscript", "object", "ol", "optgroup", "option", "output", "p",
        "param", "picture", "plaintext", "pre", "progress", "q", "rb", "rp", "rt", "rtc", "ruby", "s", "samp",
        "script", "section", "select", "set", "shadow", "slot", "small", "source", "spacer", "span", "strike",
        "strong", "style", "sub", "summary", "sup", "svg", "table", "tbody", "td", "template", "textarea",
        "tfoot", "th", "thead", "time", "title", "tr", "track", "tt", "u", "ul", "var", "video", "wbr", "xmp"}
    result = 0
    for value in request_dict.values():
        words = re.split(split_pattern, value)
        result += sum(count_appearances(words, html_tags).values())
    return result


def count_unix_shell_keywords(request_dict, split_pattern='\\W'):
    unix_keywords = {
        "bash", "cat", "csh", "dash", "du", "echo", "grep", "less", "ls", "mknod", "more", "nc", "ps",
        "rbash", "sh", "sleep", "su", "tcsh", "uname", "dev", "etc", "proc", "fd", "null", "stderr",
        "stdin", "stdout", "tcp", "udp", "zero", "group", "master.passwd", "passwd", "pwd.db", "shadow",
        "shells", "spwd.db", "self", "awk", "base64", "cat", "cc", "clang", "clang++", "curl", "diff",
        "env", "fetch", "file", "find", "ftp", "gawk", "gcc", "head", "hexdump", "id", "less", "ln",
        "mkfifo", "more", "nc", "ncat", "nice", "nmap", "perl", "php", "php5", "php7", "php-cgi", "printf",
        "psed", "python", "python2", "python3", "ruby", "sed", "socat", "tail", "tee", "telnet", "top",
        "uname", "wget", "who", "whoami", "xargs", "xxd", "yes", "bash", "curl", "ncat", "nmap", "perl",
        "php", "python", "python2", "python3", "rbash", "ruby", "wget"}
    result = 0
    for value in request_dict.values():
        words = re.split(split_pattern, value)
        result += sum(count_appearances(words, unix_keywords).values())
    return result


# def powershell_keywords(request_dict, split_patter)

def byte_distribution(request_dict):
    from statistics import mean, median
    from math import floor

    concatenated = bytearray()
    for key, value in request_dict.items():
        if key == 'Method' or key == 'Class':
            pass
        concatenated = concatenated + bytes(value, 'utf-8')

    unique_bytes_count = len(set(concatenated))
    return min(concatenated), max(concatenated), floor(mean(concatenated)), median(concatenated), unique_bytes_count


def count_appearances(words, needles):
    appearances = dict()
    for word in words:
        lowered_word = word.lower()
        if lowered_word in needles:
            if lowered_word not in appearances:
                appearances[lowered_word] = 1
            else:
                appearances[lowered_word] += 1
    return appearances


def read_requests(file_path):
    with open(file_path) as file:
        reader = csv.DictReader(file)
        raw_requests = list(reader)

        for raw_request in raw_requests:
            if raw_request['Method'] == 'GET':
                raw_request['Query'] = urllib.parse.unquote(raw_request['Query'])
            else:
                raw_request['Body'] = urllib.parse.unquote_plus(raw_request['Body'])
        return raw_requests


def predict_with_onnxruntime(onnx_model, x):
    from onnxruntime import InferenceSession

    session = InferenceSession(onnx_model.SerializeToString())
    input_name = session.get_inputs()[0].name
    res = session.run(None, {input_name: x.astype(np.float32)})
    return res[0]


def persist_classifier(name, target_model, input_data, scaler):
    """This persists the model after being fitted with the whole dataset, to allow deployment to production
    :param scaler: the component used for numerical scaling relative to the training set
    """

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', target_model)
    ])

    # ZipMap not supported by tract so this would enable at least a few models to be used
    options = {id(target_model): {'zipmap': False}}

    onnx_model = to_onnx(
        model=pipeline,
        X=input_data.to_numpy().astype(np.float32),
        options=options,
        target_opset=10
    )
    with open(f'{name}.onnx', "wb") as onnx_file:
        onnx_file.write(onnx_model.SerializeToString())
    # print(onnx_model)
    # print(training_data[0])
    # print(predict_with_onnxruntime(onnx_model, training_data))


def custom_features(raw_requests):
    data_points_list = []
    for raw_request in raw_requests:
        min_byte, max_byte, mean_byte, median_byte, unique_bytes = byte_distribution(raw_request)

        request_vector = np.array([
            attribute_length(raw_request),
            number_of_letters(raw_request),
            non_printable_characters(raw_request),
            entropy(raw_request),
            url_length(raw_request),
            path_non_alpha(raw_request),
            sql_keywords(raw_request),
            total_length(raw_request),
            count_html_tags(raw_request),
            count_js_keywords(raw_request),
            count_event_handlers(raw_request),
            count_unix_shell_keywords(raw_request),
            min_byte,
            max_byte,
            mean_byte,
            median_byte,
            unique_bytes,
            NORMAL_LABEL if raw_request.get('Class') == 'Normal' else ANOMALOUS_LABEL
        ])

        data_points_list.append(request_vector)

    data_points = np.asarray(data_points_list)
    dataframe = pd.DataFrame({
        'Attribute length': data_points[:, 0],
        'Number of letters': data_points[:, 1],
        'Non printable characters': data_points[:, 2],
        'Entropy': data_points[:, 3],
        'URL length': data_points[:, 4],
        'Non alphabetical chars in path': data_points[:, 5],
        'SQL keywords': data_points[:, 6],
        'Total length': data_points[:, 7],
        'Number of HTML tags': data_points[:, 8],
        'JavaScript keywords': data_points[:, 9],
        'JavaScript event handlers': data_points[:, 10],
        'Unix shell keywords': data_points[:, 11],
        'Minimum byte': data_points[:, 12],
        'Maximum byte': data_points[:, 13],
        'Mean byte': data_points[:, 14],
        'Median byte': data_points[:, 15],
        'Unique bytes': data_points[:, 16],
        'Label': data_points[:, 17],
    })
    return dataframe


def load_csic():
    return read_requests('data/csic2010/normalTrafficTraining.txt.csv') + \
           read_requests('data/csic2010/normalTrafficTest.txt.csv') + \
           read_requests('data/csic2010/anomalousTrafficTest.txt.csv')


def grid_search_best_parameters(parameter_grid, estimator_model, x, y):
    cross_validator = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    grid_search = GridSearchCV(
        estimator=estimator_model,
        param_grid=parameter_grid,
        n_jobs=-1,
        cv=cross_validator,
        scoring='accuracy',
        refit=True
    )
    grid_search.fit(x, y)
    return grid_search


def compute_indicators(y_true, y_pred):
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    # roc_auc = roc_auc_score(y_true=y_test, y_score=y_test_pred_probabilities[1])
    precision, recall, f_score, support = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='binary',
        pos_label=ANOMALOUS_LABEL
    )
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return [accuracy, precision, recall, f_score, mcc, tn, fp, fn, tp]


def logistic_regression(x_train, y_train, x_test, y_test):
    logistic_model = LogisticRegression(n_jobs=-1)

    # Grid search for best parameters
    c_values = [1000, 100, 10, 1.0, 0.1, 0.01, 0.001]
    parameter_grid = dict(
        penalty=['l2'],
        C=c_values,
        solver=['lbfgs'],
        max_iter=[100, 1000],
        multi_class=['multinomial', 'ovr'],
        class_weight=[None, 'balanced']
    ),
    grid_search = grid_search_best_parameters(parameter_grid, logistic_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def decision_tree(x_train, y_train, x_test, y_test):
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(x_train, y_train)

    parameter_grid = dict(max_depth=[None, 3], max_features=['sqrt', 'log2', None])
    grid_search = grid_search_best_parameters(parameter_grid, decision_tree_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def random_forest(x_train, y_train, x_test, y_test):
    random_forest_model = RandomForestClassifier(n_jobs=-1)

    parameter_grid = dict(n_estimators=[10, 100], max_features=['sqrt', 'log2', None])
    grid_search = grid_search_best_parameters(parameter_grid, random_forest_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def knn(x_train, y_train, x_test, y_test):
    knn_model = KNeighborsClassifier(n_jobs=-1)
    knn_model.fit(x_train, y_train)

    parameter_grid = dict(n_neighbors=[1, 2, 3], weights=['uniform', 'distance'])
    grid_search = grid_search_best_parameters(parameter_grid, knn_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def mlp(x_train, y_train, x_test, y_test):
    from sklearn.neural_network import MLPClassifier
    mlp_model = MLPClassifier()

    parameter_grid = dict(activation=['relu', 'tanh'], max_iter=[1000])
    grid_search = grid_search_best_parameters(parameter_grid, mlp_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def naive_bayes(x_train, y_train, x_test, y_test):
    bayes = GaussianNB()

    parameter_grid = dict()
    grid_search = grid_search_best_parameters(parameter_grid, bayes, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred)


def svm(x_train, y_train, x_test, y_test):
    svm_model = SVC()

    parameter_grid = dict(kernel=['rbf', 'linear'], C=[0.0001, 0.001, 0.01], probability=[False])
    grid_search = grid_search_best_parameters(parameter_grid, svm_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred)


def dummy_baseline(x_train, y_train, x_test, y_test):
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(x_train, y_train)

    y_test_pred = pd.DataFrame(dummy.predict(x_test))

    return compute_indicators(y_true=y_test, y_pred=y_test_pred), dummy


def plot_histograms(df):
    for index, column in enumerate(df):
        if column == "Label":
            continue
        pyplot.figure(index)
        pyplot.xlabel(column)
        df[column].plot.hist(bins=10)
    pyplot.show()


def pca(x_train, y_train, x_test, y_test):
    """This uses the whole data just for plotting currently. If I want to do dimension reduction i should train PCA
    on the training data, and use that fitted PCA on the test data to avoid introducing bias.
    """
    scaled_points = pd.concat([x_train, x_test])
    targets = pd.concat([y_train, y_test])
    pca_transform = PCA(n_components=2)
    transformed_points = pca_transform.fit_transform(scaled_points)

    fig, axes = pyplot.subplots()
    axes.scatter(x=transformed_points[:, 0], y=transformed_points[:, 1], c=targets)
    axes.legend()
    # pyplot.xlabel("PC1")
    # pyplot.ylabel("PC2")
    pyplot.show()


def preprocess_http_request_for_vectorization(raw_requests):
    concatenated_requests = []
    for raw_request in raw_requests:
        concatenated_requests.append(' '.join(value for value in raw_request.values() if value != 'Class'))

    request_labels = [NORMAL_LABEL if raw_request.get('Class') == 'Normal' else ANOMALOUS_LABEL for raw_request in
                      raw_requests]
    return pd.DataFrame({'Text': concatenated_requests, 'Label': request_labels})


def text_analysis_pipeline(classifier):
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    text_pipeline = Pipeline(
        [
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', classifier)
        ])
    return text_pipeline


def bag_of_words(requests, classifier):
    dataset = preprocess_http_request_for_vectorization(requests)
    x_train, y_train, x_test, y_test = split_train_test(dataset, 0.8)
    # Coerce to arrays
    x_train = x_train['Text']
    x_test = x_test['Text']
    text_classifier = text_analysis_pipeline(classifier)
    text_classifier.fit(x_train, y_train)
    y_pred = text_classifier.predict(x_test)
    from sklearn.metrics import classification_report
    print(classification_report(y_true=y_test, y_pred=y_pred, digits=4))


def split_train_test(dataset, train_proportion):
    train, test = np.split(dataset.sample(frac=1), [int(train_proportion * len(dataset))])
    y_train = train['Label']
    x_train = train.drop(['Label'], axis=1)
    y_test = test['Label']
    x_test = test.drop(['Label'], axis=1)
    return x_train, y_train, x_test, y_test


def main():
    requests = load_csic()

    bag_of_words(requests, LogisticRegression())
    bag_of_words(requests, MultinomialNB())

    df = custom_features(requests)
    # plot_histograms(df)
    x_train, y_train, x_test, y_test = split_train_test(df, 0.8)
    # Scale the values based on the training data
    scaler = preprocessing.StandardScaler().fit(x_train[x_train.columns])
    x_train[x_train.columns] = scaler.transform(x_train[x_train.columns])
    x_test[x_train.columns] = scaler.transform(x_test[x_test.columns])

    # Do principal component analysis on whole dataset for plotting purposes (visualizing whole data)
    # pca(x_train, y_train, x_test, y_test)

    x_dataset = pd.concat([x_train, x_test])
    y_dataset = pd.concat([y_train, y_test])

    baseline_results = dummy_baseline(x_train, y_train, x_test, y_test)
    print(f'Baseline results: {baseline_results}')
    print()

    best_logistic_regression_indicators, logistic_grid_results = logistic_regression(x_train, y_train, x_test, y_test)
    print(f'Logistic regression: {best_logistic_regression_indicators}')
    print("Best params: ", logistic_grid_results.best_params_)
    print()
    final_logistic_regression = logistic_grid_results.best_estimator_.fit(x_dataset, y_dataset)
    persist_classifier("logistic_regression", final_logistic_regression, x_dataset, scaler)

    decision_tree_performance_indicators, decision_tree_grid_results = decision_tree(x_train, y_train, x_test, y_test)
    print(f'Decision tree: {decision_tree_performance_indicators}')
    print("Best params: ", decision_tree_grid_results.best_params_)
    print()
    final_decision_tree_classifier = decision_tree_grid_results.best_estimator_.fit(x_dataset, y_dataset)
    persist_classifier("decision_tree", final_decision_tree_classifier, x_dataset, scaler)

    random_forest_performance_indicators, random_forest_grid_results = random_forest(x_train, y_train, x_test, y_test)
    print(f'Random forest: {random_forest_performance_indicators}')
    print("Best params: ", random_forest_grid_results.best_params_)
    print()
    final_random_forest_classifier = random_forest_grid_results.best_estimator_.fit(x_dataset, y_dataset)
    persist_classifier("random_forest", final_random_forest_classifier, x_dataset, scaler)

    knn_performance_indicators, knn_grid_results = knn(x_train, y_train, x_test, y_test)
    print(f'kNN: {knn_performance_indicators}')
    print("Best params: ", knn_grid_results.best_params_)
    print()
    final_knn_classifier = knn_grid_results.best_estimator_.fit(x_dataset, y_dataset)
    persist_classifier("knn", final_knn_classifier, x_dataset, scaler)

    mlp_performance_indicators, mlp_grid_results = mlp(x_train, y_train, x_test, y_test)
    print(f'MLP: {mlp_performance_indicators}')
    print("Best params: ", mlp_grid_results.best_params_)
    print()
    final_mlp_classifier = mlp_grid_results.best_estimator_.fit(x_dataset, y_dataset)
    persist_classifier("mlp", final_mlp_classifier, x_dataset, scaler)

    bayes_performance_indicators = naive_bayes(x_train, y_train, x_test, y_test)
    print(f'Naive Bayes: {bayes_performance_indicators}')
    print()

    svm_performance_indicators = svm(x_train, y_train, x_test, y_test)
    print(f'SVM: {svm_performance_indicators}')
    print()

    metrics_headers = ["accuracy", "precision", "recall", "f_score", "mcc", "tn", "fp", "fn", "tp"]
    metrics_df = pd.DataFrame(
        np.array(
            [
                best_logistic_regression_indicators,
                decision_tree_performance_indicators,
                random_forest_performance_indicators,
                knn_performance_indicators,
                mlp_performance_indicators,
                bayes_performance_indicators,
                svm_performance_indicators,
            ]
        ), columns=metrics_headers)

    metrics_df.to_csv(f'results-{datetime.datetime.now().replace(microsecond=0).isoformat()}.csv', index=False)


if __name__ == "__main__":
    main()
