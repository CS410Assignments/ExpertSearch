import os
import sys

def split(filehandler, delimiter=',', row_limit=10000,
          output_name_template='output_%s.csv', output_path='.', keep_headers=True):
    """
    Splits a CSV file into multiple pieces.

    A quick bastardization of the Python CSV library.

    Arguments:

        `row_limit`: The number of rows you want in each output file. 10,000 by default.
        `output_name_template`: A %s-style template for the numbered output files.
        `output_path`: Where to stick the output files.
        `keep_headers`: Whether or not to print the headers in each output file.

    Example usage:

        >> from toolbox import csv_splitter;
        >> csv_splitter.split(open('/home/ben/input.csv', 'r'));

    """
    row_limit = int(row_limit)
    import csv
    reader = csv.reader(filehandler, delimiter=delimiter)

    current_piece = 1
    current_out_path = os.path.join(
        output_path,
        output_name_template % current_piece
    )
    print(current_out_path)
    current_out_writer = csv.writer(open(current_out_path, 'w',newline='',encoding='utf-8'), delimiter=delimiter)
    current_limit = row_limit
    if keep_headers:
        headers = next(reader)
        current_out_writer.writerow(headers)
    seen = []
    for i, row in enumerate(reader):
        if len(row) > 0:
            if (row[1] != ''):
                #print(row)
                if row[1] not in seen:
                    seen.append(row[1])
                    if i + 1 > current_limit:
                        current_piece += 1
                        current_limit = row_limit * current_piece
                        current_out_path = os.path.join(
                            output_path,
                            output_name_template % current_piece
                        )
                        current_out_writer = csv.writer(open(current_out_path, 'w',newline='',encoding='utf-8'), delimiter=delimiter)
                        if keep_headers:
                            current_out_writer.writerow(headers)
                    current_out_writer.writerow(row)


split(filehandler = open("UTexas.csv",'r',encoding='utf-8'), row_limit = 10000);
