import csv
import itertools
import os
import shutil

train_in = 'train_cluster.tsv'
test_in = 'test.tsv'
dev_in = 'dev.tsv'
metadata_out = 'metadata.csv'

if os.path.exists("metadata_out"):
    os.remove(metadata_out)
if os.path.exists("data"):
    shutil.rmtree("data")
if os.path.exists("train"):
    shutil.rmtree("train")
if os.path.exists("test"):
    shutil.rmtree("test")
if os.path.exists("dev"):
    shutil.rmtree("dev")

with open(train_in, "r") as f_train, \
     open(test_in, "r") as f_test, \
     open(dev_in, "r") as f_dev, \
     open(metadata_out, "w") as f_metadata:

    train_reader = csv.reader(f_train, delimiter='\t')
    test_reader = csv.reader(f_test, delimiter='\t')
    dev_reader = csv.reader(f_dev, delimiter='\t')

    if not os.path.exists(os.path.join("data", "train")):
        os.makedirs(os.path.join("data", "train"))
    line_count = 0
    for row in train_reader:
        if line_count == 0:
            line_count += 1
            row.insert(0, 'file_name')
            csv.writer(f_metadata).writerow(row)
        else:
            line_count += 1
            clip_name = row[1] + '.mp3'
            file_name = os.path.join('data', 'train', clip_name)
            row.insert(0, file_name)
            csv.writer(f_metadata).writerow(row)
            shutil.copyfile(os.path.join('clips', clip_name), file_name)
            print(f'Processed {line_count} lines.')

    if not os.path.exists(os.path.join("data", "test")):
        os.makedirs(os.path.join("data", "test"))
    line_count = 0
    for row in test_reader:
        if line_count == 0:
            line_count += 1
        else:
            line_count += 1
            clip_name = row[1] + '.mp3'
            file_name = os.path.join('data', 'test', clip_name)
            row.insert(0, file_name)
            csv.writer(f_metadata).writerow(row)
            shutil.copyfile(os.path.join('clips', clip_name), file_name)
            print(f'Processed {line_count} lines.')

    if not os.path.exists(os.path.join("data", "dev")):
        os.makedirs(os.path.join("data", "dev"))
    line_count = 0
    for row in dev_reader:
        if line_count == 0:
            line_count += 1
        else:
            line_count += 1
            clip_name = row[1] + '.mp3'
            file_name = os.path.join('data', 'dev', clip_name)
            row.insert(0, file_name)
            csv.writer(f_metadata).writerow(row)
            shutil.copyfile(os.path.join('clips', clip_name), file_name)
            print(f'Processed {line_count} lines.')
