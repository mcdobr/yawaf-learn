# One class classification prototype for learning
# Tutorials followed:
# One class classification https://machinelearningmastery.com/one-class-classification-algorithms/
# Multiple models and pandas usage https://lukesingham.com/whos-going-to-leave-next/
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support, matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM, SVC
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


def read_data_points(file_path):
    with open(file_path) as file:
        reader = csv.DictReader(file)
        raw_requests = list(reader)

        for raw_request in raw_requests:
            if raw_request['Method'] == 'GET':
                raw_request['Query'] = urllib.parse.unquote(raw_request['Query'])
            else:
                raw_request['Body'] = urllib.parse.unquote_plus(raw_request['Body'])

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
        return data_points


def predict_with_onnxruntime(onnx_model, x):
    from onnxruntime import InferenceSession

    session = InferenceSession(onnx_model.SerializeToString())
    input_name = session.get_inputs()[0].name
    res = session.run(None, {input_name: x.astype(np.float32)})
    return res[0]


def persist_classifier(target_model, training_data):
    onnx_model = to_onnx(model=target_model, X=training_data.astype(np.float32), target_opset=10)
    print(onnx_model)
    print(training_data[0])
    print(predict_with_onnxruntime(onnx_model, training_data))

    with open("occ_svm.onnx", "wb") as onnx_file:
        onnx_file.write(onnx_model.SerializeToString())


def load_data():
    data_points = np.concatenate(
        (
            read_data_points('data/csic2010/normalTrafficTraining.txt.csv'),
            read_data_points('data/csic2010/normalTrafficTest.txt.csv'),
            read_data_points('data/csic2010/anomalousTrafficTest.txt.csv'),
        )
    )
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
    logistic_model.fit(x_train, y_train)

    y_train_pred = logistic_model.predict(x_train)
    train_accuracy = accuracy_score(y_true=y_train, y_pred=y_train_pred)

    y_test_pred = pd.DataFrame(logistic_model.predict(x_test))

    return compute_indicators(y_true=y_test, y_pred=y_test_pred)


def decision_tree(x_train, y_train, x_test, y_test):
    decision_tree = DecisionTreeClassifier(max_depth=3)
    decision_tree.fit(x_train, y_train)

    y_test_pred = pd.DataFrame(decision_tree.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred)


def random_forest(x_train, y_train, x_test, y_test):
    random_forest = RandomForestClassifier(n_jobs=-1)
    random_forest.fit(x_train, y_train)

    y_test_pred = pd.DataFrame(random_forest.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred)


def knn(x_train, y_train, x_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=4, n_jobs=-1)
    knn.fit(x_train, y_train)

    y_test_pred = pd.DataFrame(knn.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred)


def naive_bayes(x_train, y_train, x_test, y_test):
    bayes = GaussianNB()
    bayes.fit(x_train, y_train)

    y_test_pred = pd.DataFrame(bayes.predict(x_test))
    return compute_indicators(y_true=y_test, y_pred=y_test_pred)


def svm(x_train, y_train, x_test, y_test):
    svm_model = SVC(C=1, probability=True)
    svm_model.fit(x_train, y_train)

    # train_accuracy = accuracy_score

    y_test_pred = pd.DataFrame(svm_model.predict(x_test))
    y_test_pred_probabilities = pd.DataFrame(svm_model.predict_proba(x_test))

    return compute_indicators(y_true=y_test, y_pred=y_test_pred)


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


df = load_data()

# plot_histograms(df)
train, test = np.split(df.sample(frac=1), [int(0.8 * len(df))])

print(train.shape, test.shape)
y_train = train['Label']
x_train = train.drop(['Label'], axis=1)
y_test = test['Label']
x_test = test.drop(['Label'], axis=1)
# Scale the values based on the training data
scaler = preprocessing.StandardScaler().fit(x_train[x_train.columns])
x_train[x_train.columns] = scaler.transform(x_train[x_train.columns])
x_test[x_train.columns] = scaler.transform(x_test[x_test.columns])


# Do principal component analysis on whole dataset for plotting purposes (visualizing whole data)
pca(x_train, y_train, x_test, y_test)

logistic_regression_indicators = logistic_regression(x_train, y_train, x_test, y_test)
print(f'Logistic regression: {logistic_regression_indicators}')

decision_tree_performance_indicators = decision_tree(x_train, y_train, x_test, y_test)
print(f'Decision tree: {decision_tree_performance_indicators}')

random_forest_performance_indicators = random_forest(x_train, y_train, x_test, y_test)
print(f'Random forest: {random_forest_performance_indicators}')

knn_performance_indicators = knn(x_train, y_train, x_test, y_test)
print(f'kNN: {knn_performance_indicators}')

# svm_performance_indicators = svm(x_train, y_train, x_test, y_test)
# print(f'SVM: {svm_performance_indicators}')
#
# bayes_performance_indicators = naive_bayes(x_train, y_train, x_test, y_test)
# print(f'Naive Bayes: {bayes_performance_indicators}')


metrics_headers = ["accuracy", "precision", "recall", "f_score", "mcc", "tn", "fp", "fn", "tp"]
metrics_df = pd.DataFrame(
    np.array(
        [
            logistic_regression_indicators,
            decision_tree_performance_indicators,
            random_forest_performance_indicators,
            knn_performance_indicators,
            # svm_performance_indicators,
            # bayes_performance_indicators,
        ]
    ), columns=metrics_headers)

to_csv = metrics_df.to_csv(f'results-{datetime.datetime.now().replace(microsecond=0).isoformat()}.csv', index=False)

# dataset_labels = np.data_points((train_labels, test_labels))
# nu_values = [0.015625]
# gamma_values = [0.25]
# nu_values = [0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1]
# gamma_values = [2 ** -14, 2 ** -12, 2 ** -10, 2 ** -8, 2 ** -6, 2 ** -4, 0.25, 1, 4, 16, 64, 128]

# Make the outliers -1
# train_labels[train_labels == 1] = -1
# train_labels[train_labels == 0] = 1

# test_labels[test_labels == 1] = -1
# test_labels[test_labels == 0] = 1

# for nu in nu_values:
#     for gamma in gamma_values:
#         model = evaluate_occ_svm(gamma_param=gamma, nu_param=nu)
#         model = evaluate_svm()
# persist_classifier(model, number_of_features, train_request_data_points)


# def evaluate_occ_svm(gamma_param, nu_param):
#     # create an outlier detection model
#     model = OneClassSVM(gamma=gamma_param, nu=nu_param)
#     # fit on overwhelming majority class (we can more easily generate legitimate traffic),
#     # 0 means inlier, 1 means outlier
#     model.fit(train_request_data_points)
#
#     # evaluate on the training dataset
#     train_labels_predictor = model.predict(train_request_data_points)
#     train_accuracy = accuracy_score(y_true=train_labels, y_pred=train_labels_predictor)
#
#     # detect outliers in the test set
#     test_labels_predictor = model.predict(test_request_data_points)
#
#     # Outliers are marked with -1
#     precision, recall, f_score, support = precision_recall_fscore_support(
#         y_true=test_labels,
#         y_pred=test_labels_predictor,
#         average='binary',
#         pos_label=ANOMALOUS_LABEL
#     )
#
#     mcc = matthews_corrcoef(y_true=test_labels, y_pred=test_labels_predictor)
#     test_accuracy = accuracy_score(y_true=test_labels, y_pred=test_labels_predictor)
#
#     print('Nu = ', nu_param)
#     print('Gamma = ', gamma_param)
#     print('Precision: ', precision)
#     print('Recall: ', recall)
#     print('F-score: ', f_score)
#     print('Support: ', support)
#
#     print('Train Accuracy ', train_accuracy)
#     print('Test Accuracy: ', test_accuracy)
#     print('MCC: ', mcc)
#     tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
#     print(f'TP = {tp}, FP={fp}, TN={tn}, FN={fn}')
#     print('True positive rate:', tp / (tp + fn))
#     print('False positive rate: ', fp / (fp + tn))
#     return model