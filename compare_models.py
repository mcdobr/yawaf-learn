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
import sys
import urllib.parse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from skl2onnx import to_onnx
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef, accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
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
            raw_request['Path'] = urllib.parse.unquote(raw_request['Path'])
            if raw_request['Method'] == 'GET':
                raw_request['Query'] = urllib.parse.unquote_plus(raw_request['Query'])
            else:
                raw_request['Body'] = urllib.parse.unquote_plus(raw_request['Body'])
        return raw_requests


def predict_with_onnxruntime(onnx_model, x):
    from onnxruntime import InferenceSession

    session = InferenceSession(onnx_model.SerializeToString())
    input_name = session.get_inputs()[0].name
    res = session.run(None, {input_name: x.astype(np.float32)})
    return res[0]


def create_custom_features_pipeline(scaler, classifier):
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', classifier)
    ])
    return pipeline


def create_persisted_model(name, pipeline, x, y):
    """This persists the model after being fitted with the whole dataset, to allow deployment to production
    :param y:
    """
    from onnxconverter_common import StringTensorType

    # ZipMap not supported by tract so this would enable at least a few models to be used
    pipeline.named_steps['classifier'].fit(x, y)
    options = {id(pipeline.named_steps['classifier']): {'zipmap': False}}

    if isinstance(x, list) and isinstance(x[0], dict):
        onnx_model = to_onnx(
            model=pipeline,
            initial_types=[("raw_request", StringTensorType([None, 1]))],
            options=options,
            target_opset=10
        )
    else:
        onnx_model = to_onnx(
            model=pipeline,
            X=x.to_numpy().astype(np.float32),
            options=options,
            target_opset=10
        )
    with open(f'{name}.onnx', "wb") as onnx_file:
        onnx_file.write(onnx_model.SerializeToString())
    return onnx_model


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
        # 'Non printable characters': data_points[:, 2],
        'Entropy': data_points[:, 3],
        'URL length': data_points[:, 4],
        'Non alphabetical chars in path': data_points[:, 5],
        'SQL keywords': data_points[:, 6],
        'Total length': data_points[:, 7],
        'Number of HTML tags': data_points[:, 8],
        'JavaScript keywords': data_points[:, 9],
        # 'JavaScript event handlers': data_points[:, 10],
        'Unix shell keywords': data_points[:, 11],
        # 'Minimum byte': data_points[:, 12],
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
    cross_validator = RepeatedStratifiedKFold(n_splits=2, n_repeats=3)
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
        average='weighted',
    )
    mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    fpr = fp / (fp + tn)
    return [accuracy, precision, recall, f_score, mcc, fpr, tn, fp, fn, tp]


# Logistic regression: [0.8282079019213929, 0.741652518392756, 0.519730319254412, 0.6111694065524075, 0.5184163742468525, 13457, 913, 2422, 2621]
# Best params:  {'classifier__C': 10, 'classifier__class_weight': None, 'classifier__max_iter': 100, 'classifier__multi_class': 'ovr', 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs', 'selector__k': 'all'}
def logistic_regression(x_train, y_train, x_test, y_test):
    logistic_model = LogisticRegression(n_jobs=-1)

    # Grid search for best parameters
    c_values = [10, 1.0, 0.1]
    parameter_grid = {
        'selector__k': [3, 5, 10, 'all'],
        'classifier__penalty': ['l2'],
        'classifier__C': c_values,
        'classifier__solver': ['lbfgs'],
        'classifier__max_iter': [100],
        'classifier__multi_class': ['ovr'],
        'classifier__class_weight': [None, 'balanced']
    }

    logistic_pipeline = Pipeline([
        ('selector', SelectKBest(score_func=f_classif)),
        ('classifier', logistic_model)
    ])
    grid_search = grid_search_best_parameters(parameter_grid, logistic_pipeline, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


# Decision tree: [0.9577602637407923, 0.9245928011260809, 0.9117588736862978, 0.9181309904153355, 0.8897099864788609, 13995, 375, 445, 4598]
# Best params:  {'max_depth': None, 'max_features': None}
def decision_tree(x_train, y_train, x_test, y_test):
    decision_tree_model = DecisionTreeClassifier()
    decision_tree_model.fit(x_train, y_train)

    initial_parameter_grid = dict(
        max_depth=[None, 3],
        max_features=['sqrt', None],
    )
    best_parameter_grid = dict(
        max_depth=[None],
        max_features=[None],
    )
    grid_search = grid_search_best_parameters(best_parameter_grid, decision_tree_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def random_forest(x_train, y_train, x_test, y_test):
    random_forest_model = RandomForestClassifier(n_jobs=-1)

    # Already tried class_weight = 'balanced' / 'balanced_subsample and None always wins
    initial_parameter_grid = dict(
        n_estimators=[10, 100, 1000],
        max_features=[None, 'sqrt'],
        class_weight=[None, 'balanced', 'balanced_subsample']
    )
    best_parameter_grid = dict(
        n_estimators=[1000],
        max_features=[None],
        class_weight=[None]
    )

    grid_search = grid_search_best_parameters(best_parameter_grid, random_forest_model, x_train, y_train)
    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def extra_trees(x_train, y_train, x_test, y_test):
    from sklearn.ensemble import ExtraTreesClassifier
    extra_forest = ExtraTreesClassifier()

    best_parameter_grid = {
        'n_estimators': [1000]
    }

    grid_search = grid_search_best_parameters(best_parameter_grid, extra_forest, x_train, y_train)
    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def knn(x_train, y_train, x_test, y_test):
    knn_model = KNeighborsClassifier(n_jobs=-1)
    knn_model.fit(x_train, y_train)

    parameter_grid = dict(n_neighbors=[3, 5], weights=['uniform', 'distance'])
    grid_search = grid_search_best_parameters(parameter_grid, knn_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def mlp(x_train, y_train, x_test, y_test):
    from sklearn.neural_network import MLPClassifier
    mlp_model = MLPClassifier()

    parameter_grid = dict(
        # /*(50, 50, 50), (50, 100, 50),
        hidden_layer_sizes=[(100,)],
        alpha=[0.0001],  # 0.01,
        activation=['relu'],  # tanh
        max_iter=[1000],
        learning_rate=['constant']  # adaptive
    )
    grid_search = grid_search_best_parameters(parameter_grid, mlp_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def naive_bayes(x_train, y_train, x_test, y_test):
    bayes = GaussianNB()

    parameter_grid = dict()
    grid_search = grid_search_best_parameters(parameter_grid, bayes, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def linear_svm(x_train, y_train, x_test, y_test):
    from sklearn.svm import LinearSVC
    svm_model = LinearSVC()

    initial_parameter_grid = dict(
        C=[0.0001, 0.001, 0.01, 0.1, 1.0],
        max_iter=[20000],
    )
    best_parameter_grid = dict(
        C=[1.0],
        max_iter=[20000]
    )
    grid_search = grid_search_best_parameters(best_parameter_grid, svm_model, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def linear_svm_with_kernel_approximation(x_train, y_train, x_test, y_test):
    from sklearn.svm import LinearSVC
    from sklearn.kernel_approximation import Nystroem
    pipeline = Pipeline(
        [
            ("kernel_sampler", Nystroem()),
            ("classifier", LinearSVC())
        ]
    )
    initial_parameter_grid = {
        'kernel_sampler__n_components': [100, 200, 300, 400, 500],
    }
    best_parameter_grid = {
        'kernel_sampler__n_components': [300]
    }
    grid_search = grid_search_best_parameters(best_parameter_grid, pipeline, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


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

    print(normal_points)
    fig, axes = plt.subplots()
    axes.scatter(
        x=anomaly_points[:, 0],
        y=anomaly_points[:, 1],
        c=['red'],
        label='Anomaly',
        edgecolors='black',
        linewidths=1,
    )
    axes.scatter(
        x=normal_points[:, 0],
        y=normal_points[:, 1],
        c=['blue'],
        label='Normal',
        edgecolors='black',
        linewidths=1,
    )

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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest
    text_pipeline = Pipeline(
        [
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
            ('feature_reduction', SelectKBest(k=10)),
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

    # x_train_numerical = text_classifier.named_steps['tfidf'].transform(x_train)
    # x_test_numerical = text_classifier.named_steps['tfidf'].transform(x_test)
    # pca(x_train_numerical, y_train, x_test_numerical, y_test)
    from sklearn.metrics import classification_report
    logging.info("Classification report: {}".format(classification_report(y_true=y_test, y_pred=y_pred, digits=4)))
    return text_classifier


def voting_ensemble(x_train, y_train, x_test, y_test):
    models = [
        ('lr', LogisticRegression()),
        ('dt', DecisionTreeClassifier()),
        ('mlp', MLPClassifier(max_iter=1000))]
    voting_model = VotingClassifier(estimators=models, n_jobs=-1)

    pipeline = Pipeline([
        ('sel', SelectKBest()),
        ('vm', voting_model)
    ])

    parameter_grid = {
        'sel__k': [5, 10, 'all'],
        'vm__voting': ['hard', 'soft'],
        'vm__weights': [[0.1, 0.7, 0.2], [1, 1, 1]]
    }
    grid_search = grid_search_best_parameters(parameter_grid, pipeline, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def bagging_knn(x_train, y_train, x_test, y_test):
    bagging = BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=1, n_jobs=-1),
                                bootstrap=False,
                                n_estimators=10,
                                max_features=5,
                                n_jobs=-1)

    parameter_grid = {}
    grid_search = grid_search_best_parameters(parameter_grid, bagging, x_train, y_train)
    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


def arbitrary_stack(x_train, y_train, x_test, y_test):
    models = [
        ('dt', DecisionTreeClassifier(max_features=None, max_depth=None)),
        ('mlp', MLPClassifier(max_iter=1000)),
        ('knn', KNeighborsClassifier(n_neighbors=5, n_jobs=-1))
    ]
    stack = StackingClassifier(
        estimators=models,
        final_estimator=None,  # Logistic regression
        n_jobs=-1)
    parameter_grid = dict(
    )
    grid_search = grid_search_best_parameters(parameter_grid, stack, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


# [0.8809045484984289, 0.9859571322985957, 0.5398624038850668, 0.6976987447698745, 0.6756985555201576, 14433, 38, 2274, 2668]
def arbitrary_disjoint_subspace_voting_ensemble(x_train, y_train, x_test, y_test):
    sorted_features = get_random_forest_feature_importance(x_train, y_train)

    first_subset_features = list(map(lambda x: x[1], sorted_features[:5]))
    second_subset_features = list(map(lambda x: x[1], sorted_features[5:10]))
    third_subset_features = list(map(lambda x: x[1], sorted_features[10:]))

    x_train_first = x_train[first_subset_features]
    x_test_first = x_test[first_subset_features]

    x_train_second = x_train[second_subset_features]
    x_test_second = x_test[second_subset_features]

    x_train_third = x_train[third_subset_features]
    x_test_third = x_test[third_subset_features]

    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(x_train_first, y_train)
    y_pred_first = rf.predict(x_test_first)
    logging.info("First feature subset: {}".format(compute_indicators(y_true=y_test, y_pred=y_pred_first)))

    mlp = MLPClassifier(max_iter=1000)
    mlp.fit(x_train_second, y_train)
    y_pred_second = mlp.predict(x_test_second)
    logging.info("Second feature subset: {}".format(compute_indicators(y_true=y_test, y_pred=y_pred_second)))

    knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    knn.fit(x_train_third, y_train)
    y_pred_third = knn.predict(x_test_third)
    logging.info("Third feature subset: {}".format(compute_indicators(y_true=y_test, y_pred=y_pred_third)))

    y_pred = [0 if sum(x) <= 1 else 1 for x in zip(y_pred_first, y_pred_second, y_pred_third)]
    return compute_indicators(y_true=y_test, y_pred=y_pred)


def split_train_test(dataset, train_proportion):
    train, test = np.split(dataset.sample(frac=1), [int(train_proportion * len(dataset))])
    y_train = train['Label']
    x_train = train.drop(['Label'], axis=1)
    y_test = test['Label']
    x_test = test.drop(['Label'], axis=1)
    return x_train, y_train, x_test, y_test


def sgd(x_train, y_train, x_test, y_test):
    sgd = SGDClassifier(n_jobs=-1)
    sgd.fit(x_train, y_train)

    parameter_grid = dict(
    )
    grid_search = grid_search_best_parameters(parameter_grid, sgd, x_train, y_train)

    y_test_pred = pd.DataFrame(grid_search.best_estimator_.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred), grid_search


# See https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_model_cv_performance(estimator, title, X, y, axes=None, ylim=None, cv=None,
                              n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    from sklearn.model_selection import learning_curve
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Number of training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Scalability (number of samples and fit times)
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Number of training examples")
    axes[1].set_ylabel("Fit times")
    axes[1].set_title("Scalability of the model")

    # Performance of the model (Fit times and score)
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.show()


def main():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(funcName)s:  %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler("learn.log"),
            logging.StreamHandler(sys.stdout),
        ]
    )

    logging.info("Starting analysis...")
    requests = load_csic()

    # tfidf_logistic_regression = bag_of_words(requests, LogisticRegression())
    # persist_classifier("tfidf_logistic_regression", tfidf_logistic_regression, requests)

    # tfidf_multinomial_naive_bayes = bag_of_words(requests, MultinomialNB())
    # persist_classifier("tfidf_multinomial_naive_bayes", tfidf_multinomial_naive_bayes, requests)

    df = custom_features(requests)
    # plot_heatmap(df)

    # plot_histograms(df)
    x_train, y_train, x_test, y_test = split_train_test(df, 0.8)

    get_random_forest_feature_importance(x_train, y_train)

    # Scale the values based on the training data
    # todo: maybe put the scaler in individual pipelines?
    scaler = preprocessing.StandardScaler().fit(x_train[x_train.columns])
    x_train[x_train.columns] = scaler.transform(x_train[x_train.columns])
    x_test[x_train.columns] = scaler.transform(x_test[x_test.columns])

    # Do principal component analysis on whole dataset for plotting purposes (visualizing whole data)
    # pca(x_train, y_train, x_test, y_test)

    x_dataset = pd.concat([x_train, x_test])
    y_dataset = pd.concat([y_train, y_test])

    logging.info("Dataset description {}".format(x_dataset.describe()))

    baseline_results = dummy_baseline(x_train, y_train, x_test, y_test)
    logging.info("Baseline results: {}".format(baseline_results))

    best_logistic_regression_indicators, logistic_grid_results = logistic_regression(x_train, y_train, x_test, y_test)
    save_test_performance("Logistic regression", logistic_grid_results, best_logistic_regression_indicators, x_train,
                          y_train)
    create_persisted_model("logistic_regression",
                           create_custom_features_pipeline(scaler=scaler,
                                                           classifier=logistic_grid_results.best_estimator_),
                           x_dataset,
                           y_dataset)

    best_linear_svm_indicators, linear_svm_grid_results = linear_svm(x_train, y_train, x_test, y_test)
    save_test_performance("Linear SVM", linear_svm_grid_results, best_linear_svm_indicators, x_train, y_train)

    best_linear_svm_with_kernel_approximation_indicators, best_linear_svm_with_kernel_approximation_grid_results = linear_svm_with_kernel_approximation(
        x_train, y_train, x_test, y_test)
    save_test_performance("Linear SVM with kernel approximation",
                          best_linear_svm_with_kernel_approximation_grid_results,
                          best_linear_svm_with_kernel_approximation_indicators, x_train, y_train)

    decision_tree_performance_indicators, decision_tree_grid_results = decision_tree(x_train, y_train, x_test, y_test)
    save_test_performance("Decision Tree", decision_tree_grid_results, decision_tree_performance_indicators, x_train,
                          y_train)
    create_persisted_model("decision_tree", create_custom_features_pipeline(scaler=scaler,
                                                                            classifier=decision_tree_grid_results.best_estimator_),
                           x_dataset, y_dataset)

    random_forest_performance_indicators, random_forest_grid_results = random_forest(x_train, y_train, x_test, y_test)
    save_test_performance("Random Forest", random_forest_grid_results, random_forest_performance_indicators, x_train,
                          y_train)
    create_persisted_model("random_forest",
                           create_custom_features_pipeline(scaler=scaler,
                                                           classifier=random_forest_grid_results.best_estimator_),
                           x_dataset, y_dataset)

    extra_trees_performance_indicators, extra_trees_grid = random_forest(x_train, y_train, x_test, y_test)
    save_test_performance("Extra trees", extra_trees_grid, extra_trees_performance_indicators, x_train,
                          y_train)

    mlp_performance_indicators, mlp_grid_results = mlp(x_train, y_train, x_test, y_test)
    save_test_performance("Multi layer perceptron", mlp_grid_results, mlp_performance_indicators, x_train, y_train)
    create_persisted_model("mlp",
                           create_custom_features_pipeline(scaler=scaler, classifier=mlp_grid_results.best_estimator_),
                           x_train, y_train)

    knn_performance_indicators, knn_grid_results = knn(x_train, y_train, x_test, y_test)
    save_test_performance("K nearest neighbors", knn_grid_results, knn_performance_indicators, x_train, y_train)
    create_persisted_model("knn",
                           create_custom_features_pipeline(scaler=scaler, classifier=knn_grid_results.best_estimator_),
                           x_dataset, y_dataset)

    sgd_performance_indicators, sgd_grid_results = sgd(x_train, y_train, x_test, y_test)
    save_test_performance("Stochastic gradient descent classifier", sgd_grid_results, sgd_performance_indicators,
                          x_train, y_train)
    create_persisted_model("sgd",
                           create_custom_features_pipeline(scaler=scaler, classifier=sgd_grid_results.best_estimator_),
                           x_dataset, y_dataset)

    disjoint_subspace_voting_perf_results = arbitrary_disjoint_subspace_voting_ensemble(x_train, y_train, x_test, y_test)
    save_test_performance("Disjoint subspace voting ensemble (kNN, RF, MLP)", None, disjoint_subspace_voting_perf_results, x_train, y_train)

    bagging_knn_perf_indicators, bagging_knn_grid_results = bagging_knn(x_train, y_train, x_test, y_test)
    save_test_performance("Bagging kNN (k=1)", bagging_knn_grid_results, bagging_knn_perf_indicators, x_train, y_train)

    voting_ensemble_indicators, voting_ensemble_grid_results = voting_ensemble(x_train, y_train, x_test, y_test)
    save_test_performance("Voting ensemble", voting_ensemble_grid_results, voting_ensemble_indicators, x_train,
                          y_train)

    stack_perf_indicators, stack_grid_results = arbitrary_stack(x_train, y_train, x_test, y_test)
    save_test_performance("Stacked ensemble", stack_grid_results, stack_perf_indicators, x_train, y_train)

    logging.info("Writing values to CSV file")
    metrics_headers = ["name", "accuracy", "weighted precision", "weighted recall", "weighted f_score", "mcc", "fpr", "tn", "fp", "fn", "tp"]
    metrics_df = pd.DataFrame(
        np.array(
            [
                ["lr"] + best_logistic_regression_indicators,
                ["lin_svm"] + best_linear_svm_indicators,
                ["lin_svm_kernel_approx"] + best_linear_svm_with_kernel_approximation_indicators,
                ["dt"] + decision_tree_performance_indicators,
                ["rf"] + random_forest_performance_indicators,
                ["et"] + extra_trees_performance_indicators,
                ["knn"] + knn_performance_indicators,
                ["mlp"] + mlp_performance_indicators,
                ["lr_dt_mlp_voting"] + voting_ensemble_indicators,
                ["dt_mlp_knn_stacked"] + stack_perf_indicators,
            ]
        ), columns=metrics_headers)

    metrics_df.to_csv(f'results-{datetime.datetime.now().replace(microsecond=0).isoformat()}.csv', index=False)
    logging.info("Wrote values to CSV file")
    logging.info("Ended analysis...")


def save_test_performance(name, grid_results, performance_indicators, x_train, y_train):
    logging.info("{} performance indicators: {}".format(name, performance_indicators))
    if grid_results is not None:
        logging.info("Best params: {}".format(grid_results.best_params_))
        plot_model_cv_performance(estimator=grid_results.best_estimator_, title=name,
                                  X=x_train, y=y_train, ylim=(0.6, 1.01), cv=grid_results.cv, n_jobs=-1)


def get_random_forest_feature_importance(x_train, y_train):
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    sorted_features = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True)
    logging.info("Sorted features by random forest importance: {}".format(sorted_features))
    return sorted_features


def plot_heatmap(df):
    # Draw the heatmap with the mask and correct aspect ratio
    sns.set_theme(style="white")

    correlation = df.corr()
    mask = np.tril(np.ones_like(correlation, dtype=bool))

    f, ax = plt.subplots(figsize=(10, 10))

    color_map = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(correlation, mask=mask, cmap=color_map, vmax=.3, center=0,
                square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()
    logging.info("Plotted heatmap")


if __name__ == "__main__":
    main()
