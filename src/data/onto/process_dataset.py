import os
import sys
from sklearn.model_selection import train_test_split

dirname = os.path.dirname(__file__)  # NOQA: E402
sys.path.append(os.path.join(dirname, '../'))  # NOQA: E402

from make_dataset import parse_file


def write_array_to_iob2_file(document, file_path, separator=' ', word_ending='\n'):
    with open(file_path, 'w+') as out:
        for sentence in document:
            for (word, tag) in zip(sentence['words'], sentence['tags']):
                out.write(word + separator + tag + word_ending)

            out.write('\n')


def train_test_dev_split(data):
    X_train_dev, X_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
    X_train, X_dev = train_test_split(X_train_dev, test_size=0.25, shuffle=False)

    return X_train, X_test, X_dev


def process_data_for_spacy(X_cased, X_uncased):
    X_train_cased, X_test_cased, X_dev_cased = X_cased
    X_train_uncased, X_test_uncased, X_dev_uncased = X_uncased

    cased_output_loc = os.path.join(dirname, '../../../data/processed/onto5/spacy/cased/')
    uncased_output_loc = os.path.join(dirname, '../../../data/processed/onto5/spacy/uncased/')

    if not os.path.exists(cased_output_loc):
        os.makedirs(cased_output_loc)

    if not os.path.exists(uncased_output_loc):
        os.makedirs(uncased_output_loc)

    cased_train_path = os.path.join(cased_output_loc, 'train.iob2')
    cased_test_path = os.path.join(cased_output_loc, 'test.iob2')
    cased_dev_path = os.path.join(cased_output_loc, 'dev.iob2')

    uncased_train_path = os.path.join(uncased_output_loc, 'train.iob2')
    uncased_test_path = os.path.join(uncased_output_loc, 'test.iob2')
    uncased_dev_path = os.path.join(uncased_output_loc, 'dev.iob2')

    write_array_to_iob2_file(X_train_cased, cased_train_path, separator='|', word_ending=' ')
    write_array_to_iob2_file(X_test_cased, cased_test_path, separator='|', word_ending=' ')
    write_array_to_iob2_file(X_dev_cased, cased_dev_path, separator='|', word_ending=' ')

    write_array_to_iob2_file(X_train_uncased, uncased_train_path, separator='|', word_ending=' ')
    write_array_to_iob2_file(X_test_uncased, uncased_test_path, separator='|', word_ending=' ')
    write_array_to_iob2_file(X_dev_uncased, uncased_dev_path, separator='|', word_ending=' ')


def process_data_for_flair(X_cased, X_uncased):
    X_train_cased, X_test_cased, X_dev_cased = X_cased
    X_train_uncased, X_test_uncased, X_dev_uncased = X_uncased

    cased_output_loc = os.path.join(dirname, '../../../data/processed/onto5/flair/cased/')
    uncased_output_loc = os.path.join(dirname, '../../../data/processed/onto5/flair/uncased/')

    if not os.path.exists(cased_output_loc):
        os.makedirs(cased_output_loc)

    if not os.path.exists(uncased_output_loc):
        os.makedirs(uncased_output_loc)
    
    cased_train_path = os.path.join(cased_output_loc, 'train.iob2')
    cased_test_path = os.path.join(cased_output_loc, 'test.iob2')
    cased_dev_path = os.path.join(cased_output_loc, 'dev.iob2')

    uncased_train_path = os.path.join(uncased_output_loc, 'train.iob2')
    uncased_test_path = os.path.join(uncased_output_loc, 'test.iob2')
    uncased_dev_path = os.path.join(uncased_output_loc, 'dev.iob2')

    write_array_to_iob2_file(X_train_cased, cased_train_path, separator=' ', word_ending='\n')
    write_array_to_iob2_file(X_test_cased, cased_test_path, separator=' ', word_ending='\n')
    write_array_to_iob2_file(X_dev_cased, cased_dev_path, separator=' ', word_ending='\n')

    write_array_to_iob2_file(X_train_uncased, uncased_train_path, separator=' ', word_ending='\n')
    write_array_to_iob2_file(X_test_uncased, uncased_test_path, separator=' ', word_ending='\n')
    write_array_to_iob2_file(X_dev_uncased, uncased_dev_path, separator=' ', word_ending='\n')


def process_data():
    dirname = os.path.dirname(__file__)  # NOQA: E402
    input_loc = os.path.join(dirname, '../../../data/interim/onto5/')
    cased_file_path = os.path.join(input_loc, 'data_cased.iob2')
    uncased_file_path = os.path.join(input_loc, 'data_uncased.iob2')

    cased_file = parse_file(cased_file_path)
    uncased_file = parse_file(uncased_file_path)

    X_cased = train_test_dev_split(cased_file)
    X_uncased = train_test_dev_split(uncased_file)

    process_data_for_spacy(X_cased, X_uncased)
    process_data_for_flair(X_cased, X_uncased)


if __name__ == '__main__':
    process_data()