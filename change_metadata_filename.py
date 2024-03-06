import csv
import os

metadata_in = 'common_voice/metadata_old.csv'
metadata_out = 'common_voice/metadata.csv'


def convert(set):
    return sorted(set)


def change_basename(file_path, new_basename):
    # Split the file path into directory, basename, and extension
    directory, old_basename = os.path.split(file_path)
    old_basename_without_extension, extension = os.path.splitext(old_basename)

    # Create the new file path with the provided new_basename and the original extension
    new_file_path = os.path.join(directory, f"{new_basename}{extension}")

    return new_file_path


def get_basename(file_path):
    directory, basename = os.path.split(file_path)
    return basename


with open(metadata_in, 'r') as fin, \
     open(metadata_out, 'w') as fout:

    os.chdir('common_voice')

    reader = csv.reader(fin)
    writer = csv.writer(fout)

    # Skip the header row
    next(reader)

    client_id = set()

    # Iterate through the remaining rows
    for row in reader:
        client_id.add(row[1])

    print(len(client_id))
    client_id = convert(client_id)

    fin.seek(0)

    prev_client_index = 0
    prev_file_number = 0
    for line_count, row in enumerate(reader):
        if line_count > 0:
            client_index = client_id.index(row[1]) + 1
            if client_index == prev_client_index:
                file_number = prev_file_number + 1
            else:
                prev_client_index = client_index
                file_number = 1
            prev_file_number = file_number
            row[2] = get_basename(row[0])
            new_file_name = change_basename(row[0], str(client_index) + '_' + str(file_number))
            # os.rename(row[0], new_file_name)
            print(os.path.join('clips', get_basename(row[0])))
            os.rename(os.path.join('clips', get_basename(row[0])), os.path.join('clips', get_basename(new_file_name)))
            row[0] = new_file_name
            row[1] = str(client_index)

        writer.writerow(row)

        # print(change_basename(row[0], 'abc'))

