import pandas as pd

csv1 = '/home/bici/active-learning-asr/data/final/metadata.csv'
csv2 = '/home/bici/Music/active-learning-asr/data/final/metadata.csv'

# csv1 = '/home/bici/active-learning-asr/data/common_voice/metadata.csv'
# csv2 = '/home/bici/Music/active-learning-asr/data/common_voice/metadata.csv'
# csv2 = '/home/bici/active-learning-asr/data/common_voice/compare.csv'

# csv1 = '/home/bici/active-learning-asr/data/librispeech/metadata.csv'
# csv2 = '/home/bici/Music/active-learning-asr/data/librispeech/metadata.csv'

csv1 = '/home/bici/active-learning-asr/data/train_random_4/metadata.csv'
csv2 = '/home/bici/active-learning-asr/data/train_random_4/metadata.csv'

csv1 = '/home/bici/active-learning-asr/test/data/train_inverse_2/metadata.csv'
csv2 = '/home/bici/active-learning-asr/test/data/train_random_2/metadata_fmle.csv'

df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)

duplicates = df1['file_name'].duplicated().sum()
print('Duplicate num:')
print(duplicates)

merge = pd.merge(df1, df2, how='inner')
print('Number of same rows:')
print(merge.shape[0])

exit()

print(df1.shape)
print(df2.shape)
assert df1.shape == df2.shape
assert set(df1.columns) == set(df2.columns)

assert set(df1['file_name']) == set(df2['file_name'])

diff_sentence = df1['sentence']!= df2['sentence']
# print(df1[diff_sentence=True])
print(set(df1['sentence']) - set(df2['sentence']))
print(set(df2['sentence']) - set(df1['sentence']))
assert set(df1['sentence']) == set(df2['sentence'])

exit()

print('--- path ---')
diff_path = df1['path']!= df2['path']
# print(diff_path)

# equals_path = df1['path'].equals(df2['path'])
# print(equals_path)



print('--- sentence ---')
diff_sentence = df1['sentence']!= df2['sentence']
# print(diff_sentence)

diff_file_name = df1['file_name'] != df2['file_name']
print(diff_file_name)
print(diff_file_name.shape)
print(diff_file_name[diff_file_name==True])
print(df1[diff_file_name==True]['sentence'])

# equals_sentence = df1['sentence'].equals(df2['sentence'])
# print(equals_sentence)


# pd.set_option('display.max_colwidth', 100)
# print(df1[diff_sentence==True]['sentence'])
# print(df2[diff_sentence==True]['sentence'])

# assert df1.equals(df2)  # This also handles NaN values
# assert (df1 == df2).all().all()
