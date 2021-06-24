import email
import pprint
from io import StringIO
import re

input_file_name = 'data/csic/normalTrafficTraining.txt'
output_file_name = input_file_name + '.csv'

with open(input_file_name, 'rt') as input_file_name, open(output_file_name, 'wt') as output_file:
    raw_requests_text = input_file_name.read()

    raw_requests = re.split('\n\n\n', raw_requests_text)
    # pprint.pprint(raw_requests)
    header_keys = set()
    for raw_request in raw_requests:
        if not raw_request:
            continue
        request_lines = raw_request.splitlines()
        if not request_lines[-2]:
            raw_headers = request_lines[1:-3]
            body = request_lines[-1]
        else:
            raw_headers = request_lines[1:]
            body = ''
        request_line = request_lines[0]

        headers_split = [raw_header.split(':', 1) for raw_header in raw_headers]

        headers = {}
        for raw_header_split in headers_split:
            headers[raw_header_split[0]] = raw_header_split[1]

        header_keys.update(headers.keys())

        print(headers)
        # print(request_line)
        # print(headers)
        # print(body)
    print(header_keys)