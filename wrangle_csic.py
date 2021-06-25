import csv
from urllib.parse import urlparse
import re

input_file_name = 'data/csic/anomalousTrafficTest.txt'
output_file_name = input_file_name + '.csv'

with open(input_file_name, 'rt') as input_file_name, open(output_file_name, 'wt') as output_file:
    raw_requests_text = input_file_name.read()

    raw_requests = re.split('\n\n\n', raw_requests_text)

    parsed_requests = []

    properties = set()
    properties.add('Scheme')
    properties.add('Method')
    properties.add('Path')
    properties.add('Query')
    properties.add('Body')
    properties.add('Version')
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

        [http_method, url_string, http_version] = request_line.split(' ')
        url = urlparse(url_string)

        headers_split = [raw_header.split(':', 1) for raw_header in raw_headers]

        headers = {}
        for raw_header_split in headers_split:
            headers[raw_header_split[0]] = raw_header_split[1].strip()

        properties.update(headers.keys())

        parsed_request = {
            'Scheme': url.scheme,
            'Method': http_method,
            'Path': url.path,
            'Query': url.query,
            'Body': body,
            'Version': http_version
        }
        parsed_request.update(headers)
        parsed_requests.append(parsed_request)

    dictionary_writer = csv.DictWriter(output_file, properties)
    dictionary_writer.writeheader()
    for row in parsed_requests:
        dictionary_writer.writerow(row)
