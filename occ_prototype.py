# One class classification prototype for learning
# Tutorial followed from https://machinelearningmastery.com/one-class-classification-algorithms/
import csv
import re
from collections import Counter

import numpy as np
from matplotlib import pyplot
from numpy import where
from sklearn.metrics import f1_score, precision_recall_fscore_support, matthews_corrcoef
from sklearn.svm import OneClassSVM

sql2016_keywords = {
    "abs", "acos", "all", "allocate", "alter", "and", "any", "are", "array", "array_agg", "array_max_cardinality", "as",
    "asensitive", "asin", "asymmetric", "at", "atan", "atomic", "authorization", "avg", "begin", "begin_frame",
    "begin_partition", "between", "bigint", "binary", "blob", "boolean", "both", "by", "call", "called", "cardinality",
    "cascaded", "case", "cast", "ceil", "ceiling", "char", "character", "character_length", "char_length", "check",
    "classifier", "clob", "close", "coalesce", "collate", "collect", "column", "commit", "condition", "connect",
    "constraint", "contains", "convert", "copy", "corr", "corresponding", "cos", "cosh", "count", "covar_pop",
    "covar_samp", "create", "cross", "cube", "cume_dist", "current", "current_catalog", "current_date",
    "current_default_transform_group", "current_path", "current_role", "current_row", "current_schema", "current_time",
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
    "match", "matches", "match_number", "match_recognize", "max", "member", "merge", "method", "min", "minute", "mod",
    "modifies", "module", "month", "multiset", "national", "natural", "nchar", "nclob", "new", "no", "none",
    "normalize", "not", "nth_value", "ntile", "null", "nullif", "numeric", "occurrences_regex", "octet_length", "of",
    "offset", "old", "omit", "on", "one", "only", "open", "or", "order", "out", "outer", "over", "overlaps", "overlay",
    "parameter", "partition", "pattern", "per", "percent", "percentile_cont", "percentile_disc", "percent_rank",
    "period", "portion", "position", "position_regex", "power", "precedes", "precision", "prepare", "primary",
    "procedure", "ptf", "range", "rank", "reads", "real", "recursive", "ref", "references", "referencing", "regr_avgx",
    "regr_avgy", "regr_count", "regr_intercept", "regr_r2", "regr_slope", "regr_sxx", "regr_sxy", "regr_syy", "release",
    "result", "return", "returns", "revoke", "right", "rollback", "rollup", "row", "rows", "row_number", "running",
    "savepoint", "scope", "scroll", "search", "second", "seek", "select", "sensitive", "session_user", "set", "show",
    "similar", "sin", "sinh", "skip", "smallint", "some", "specific", "specifictype", "sql", "sqlexception", "sqlstate",
    "sqlwarning", "sqrt", "start", "static", "stddev_pop", "stddev_samp", "submultiset", "subset", "substring",
    "substring_regex", "succeeds", "sum", "symmetric", "system", "system_time", "system_user", "table", "tablesample",
    "tan", "tanh", "then", "time", "timestamp", "timezone_hour", "timezone_minute", "to", "trailing", "translate",
    "translate_regex", "translation", "treat", "trigger", "trim", "trim_array", "true", "truncate", "uescape", "union",
    "unique", "unknown", "unnest", "update", "upper", "user", "using", "value", "values", "value_of", "varbinary",
    "varchar", "varying", "var_pop", "var_samp", "versioning", "when", "whenever", "where", "width_bucket", "window",
    "with", "within", "without", "year"
}


def attribute_length(request_dict):
    method = request_dict.get('Method')
    if method == 'GET':
        return len(request_dict.get('Query'))
    else:
        return len(request_dict.get('Body'))


def sql_keywords(request_dict):
    method = request_dict.get('Method')
    if method == 'GET':
        query_params = request_dict.get('Query').lower()
        words = re.split('[ +&]', query_params)
    else:
        body = request_dict.get('Body').lower()
        words = re.split('[ +&]', body)
    appearances = count_appearances(words, sql2016_keywords)
    return sum(appearances.values())


def count_appearances(words, needles):
    appearances = dict()
    for word in words:
        if word in needles:
            if word not in appearances:
                appearances[word] = 1
            else:
                appearances[word] += 1
    return appearances


def read_data_points(file_path):
    with open(file_path) as file:
        reader = csv.DictReader(file)
        raw_requests = list(reader)

        data_points = np.array([
            np.array([
                attribute_length(raw_request),
                sql_keywords(raw_request),
            ])
            for raw_request in raw_requests
        ])
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
model = OneClassSVM(gamma='scale', nu=0.01)

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

print('Classes: ', test_labels)
print('Precision: ', precision)
print('Recall: ', recall)
print('F-score: ', f_score)
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
