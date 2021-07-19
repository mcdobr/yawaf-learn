# One class classification prototype for learning
# Tutorial followed from https://machinelearningmastery.com/one-class-classification-algorithms/
import csv
import re
from collections import Counter

import numpy as np
from matplotlib import pyplot
from numpy import where
from sklearn.metrics import f1_score, precision_recall_fscore_support, matthews_corrcoef, accuracy_score
from sklearn.svm import OneClassSVM


def attribute_length(request_dict):
    method = request_dict.get('Method')
    if method == 'GET':
        return len(request_dict.get('Query'))
    else:
        return len(request_dict.get('Body'))


def sql_keywords(request_dict, split_pattern='[ +&]'):
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
        query_params = request_dict.get('Query').lower()
        words = re.split(split_pattern, query_params)
    else:
        body = request_dict.get('Body').lower()
        words = re.split(split_pattern, body)
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


def count_js_keywords(request_dict, split_pattern='[ +&]'):
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
    for value in request_dict.values():
        words = re.split(split_pattern, value)
        result += sum(count_appearances(words, javascript_keywords).values())
    return result


def count_event_handlers(request_dict, split_pattern='[ +&]'):
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


def count_unix_shell_keywords(request_dict, split_pattern='[ +&]'):
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

        data_points_list = []
        for raw_request in raw_requests:
            min_byte, max_byte, mean_byte, median_byte, unique_bytes = byte_distribution(raw_request)

            request_vector = np.array([
                attribute_length(raw_request),
                sql_keywords(raw_request),
                total_length(raw_request),
                count_js_keywords(raw_request),
                count_event_handlers(raw_request),
                count_unix_shell_keywords(raw_request),
                min_byte,
                max_byte,
                mean_byte,
                median_byte,
                unique_bytes
            ])

            data_points_list.append(request_vector)

        data_points = np.asarray(data_points_list)
        labels = np.array([
            0 if raw_request.get('Class') == 'Normal' else 1 for raw_request in raw_requests
        ])
    return data_points, labels


def compute_benchmarks(y_actual, y_predictor, positive_value):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(y_predictor)):
        if y_predictor[i] == y_actual[i] and y_predictor[i] == positive_value:
            true_positive += 1
        elif y_predictor[i] == positive_value and y_predictor[i] != y_actual[i]:
            false_positive += 1
        elif y_predictor[i] != positive_value and y_predictor[i] == y_actual[i]:
            true_negative += 1
        elif y_predictor[i] != positive_value and y_predictor[i] != y_actual[i]:
            false_negative += 1
    return true_positive, false_positive, true_negative, false_negative


train_requests, train_labels = read_data_points('data/csic2010/normalTrafficTraining.txt.csv')
test_normal, test_normal_labels = read_data_points('data/csic2010/normalTrafficTest.txt.csv')
test_anomalous, test_anomalous_labels = read_data_points('data/csic2010/anomalousTrafficTest.txt.csv')

test_requests = np.concatenate((test_normal, test_anomalous))
test_labels = np.concatenate((test_normal_labels, test_anomalous_labels))

dataset_requests = np.concatenate((train_requests, test_requests))
dataset_labels = np.concatenate((train_labels, test_labels))

# print(dataset_requests)
# print(dataset_labels)

counter = Counter(dataset_labels)
# print(counter)

for label, _ in counter.items():
    row_index_x = where(dataset_labels == label)[0]
    pyplot.scatter(dataset_requests[row_index_x, 0],
                   dataset_requests[row_index_x, 1],
                   label=str(label))
pyplot.legend()
pyplot.show()

## create an outlier detection model
model = OneClassSVM(gamma='auto', nu=0.1)

# fit on overwhelming majority class (we can more easily generate legitimate traffic),
# 0 means inlier, 1 means outlier
model.fit(train_requests)

# detect outliers in the test set
label_predictor = model.predict(test_requests)

# Make the outliers -1
test_labels[test_labels == 1] = -1
test_labels[test_labels == 0] = 1

# Outliers are marked with -1
precision, recall, f_score, support = precision_recall_fscore_support(
    y_true=test_labels,
    y_pred=label_predictor,
    average='binary',
    pos_label=-1
)
mcc = matthews_corrcoef(y_true=test_labels, y_pred=label_predictor)
accuracy = accuracy_score(y_true=test_labels, y_pred=label_predictor)

print('Classes: ', test_labels)
print('Precision: ', precision)
print('Recall: ', recall)
print('F-score: ', f_score)
print('Accuracy: ', accuracy)
print('Support: ', support)
print('MCC: ', mcc)

tp, fp, tn, fn = compute_benchmarks(y_actual=test_labels, y_predictor=label_predictor, positive_value=-1)
print(tp, fp, tn, fn)
print('Manual precision: ', tp / (tp + fp))
print('Manual recall: ', tp / (tp + fn))
print('True positive rate:', tp / (tp + fn))
print('False positive rate: ', fp / (fp + tn))

test_counter = Counter(test_labels)
predictor_counter = Counter(label_predictor)

print('Actual counter', test_counter)
print('Predictor counter', predictor_counter)

from sklearn2pmml import sklearn2pmml, PMMLPipeline

pipeline = PMMLPipeline([
    ("classifier", model)
])

sklearn2pmml(pipeline, "occ_svm.pmml", with_repr=True)
