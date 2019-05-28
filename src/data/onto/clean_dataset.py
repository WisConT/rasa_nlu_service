import os
import sys
import re
# from data.make_dataset_onto import parse_file

"""
Credit: yohanesgultom
https://gist.github.com/yohanesgultom/630a831eff1fbdcd84b3cfec6feabe02
"""

START_PATTERN = re.compile(r'^(.*?)<ENAMEX$', re.I)
END_SINGLE_PATTERN = re.compile(r'^TYPE="(.*?)">(.*?)<\/ENAMEX>(.*?)$', re.I)
END_SINGLE_OTHER_ATTR_PATTERN = re.compile(r'^[A-Z_]*?="(.*?)">(.*?)<\/ENAMEX>(.*?)$', re.I)
TYPE_PATTERN = re.compile(r'^TYPE="([^"]*?)">(.*?)$', re.I)
TYPE_NO_END_PATTERN = re.compile(r'^TYPE="([^"]*?)"$', re.I)
OTHER_ATTR_PATTERN = re.compile(r'^[A-Z_]*?="([^"]*?)">(.*)$', re.I)
END_MULTI_PATTERN = re.compile(r'^(.*?)</ENAMEX>(.*?)$', re.I)
EOS_PATTERN = re.compile(r'^([^<>]*)\.?\t(\d+)$', re.I)
DOC_PATTERN = re.compile(r'<(\/)?DOC', re.I)
NON_ENTITY_TYPE = 'O'


def check_and_process_eos(out, cur_type, token, prefix='', cased=True):
    match = re.match(EOS_PATTERN, token)
    if match:
        entity_text = match.group(1) if cased else match.group(1).lower()
        out.write(entity_text + '\t' + prefix + cur_type + '\n')
        out.write('.' + '\t' + cur_type + '\n')
        out.write('\n')
        return True
    return False


def convert(file_from, file_to, cased=True):
    cur_type = NON_ENTITY_TYPE

    with open(file_from, 'r') as f, open(file_to, 'a') as out:
        out.write('-DOCUMENT-\n')
        for line in f:
            if re.match(DOC_PATTERN, line):
                continue

            for token in line.strip().split(' '):
                token = token.strip()
                if token.isspace() or token == '':
                    continue

                match = re.match(START_PATTERN, token)
                if match:
                    if match.group(1):
                        entity_text = match.group(1) if cased else match.group(1).lower()
                        out.write(entity_text + '\t' + NON_ENTITY_TYPE + '\n')
                    continue

                match = re.match(END_SINGLE_PATTERN, token)
                if match:
                    entity_text = match.group(2) if cased else match.group(2).lower()
                    out.write(entity_text + '\t' + 'B-' + match.group(1) + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(out, cur_type, match.group(3), cased=cased) and match.group(3):
                        entity_text = match.group(3) if cased else match.group(3).lower()
                        out.write(entity_text + '\t' + cur_type + '\n')
                    continue

                match = re.match(END_SINGLE_OTHER_ATTR_PATTERN, token)
                if match:
                    entity_text = match.group(2) if cased else match.group(2).lower()
                    out.write(entity_text + '\t' + 'B-' + cur_type + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(out, cur_type, match.group(3), cased=cased) and match.group(3):
                        entity_text = match.group(3) if cased else match.group(3).lower()
                        out.write(entity_text + '\t' + cur_type + '\n')
                    continue

                match = re.match(TYPE_PATTERN, token)
                if match:
                    cur_type = match.group(1)
                    entity_text = match.group(2) if cased else match.group(2).lower()
                    out.write(entity_text + '\t' + 'B-' + cur_type + '\n')
                    continue

                match = re.match(TYPE_NO_END_PATTERN, token)
                if match:
                    cur_type = match.group(1)
                    continue
                
                match = re.match(OTHER_ATTR_PATTERN, token)
                if match:
                    cur_type = match.group(1)
                    entity_text = match.group(2) if cased else match.group(2).lower()
                    out.write(entity_text + '\t' + 'B-' + cur_type + '\n')
                    continue

                match = re.match(END_MULTI_PATTERN, token)
                if match:
                    entity_text = match.group(1) if cased else match.group(1).lower()
                    out.write(entity_text + '\t' + 'I-' + cur_type + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(out, cur_type, match.group(2), cased=cased) and match.group(2):
                        entity_text = match.group(2) if cased else match.group(2).lower()
                        out.write(entity_text + '\t' + cur_type + '\n')
                    continue

                if check_and_process_eos(out, cur_type, token, 'I-', cased=cased):
                    continue

                if len(token) == 2 and token[0] == '/':
                    token = token[1]
                
                token = token if cased else token.lower()

                if cur_type != NON_ENTITY_TYPE:
                    out.write(token + '\t' + 'I-' + cur_type + '\n')
                else:
                    out.write(token + '\t' + cur_type + '\n')

            out.write('\n')


def clean_dataset():
    print("Cleaning dataset...")
    dirname = os.path.dirname(__file__)  # NOQA: E402
    external_data = os.path.join(dirname, '../../../data/external/onto5/english/annotations/')
    directory_paths = [(dirpath, list(filter(lambda x: x.endswith('.name'), filenames))) for (dirpath, dirnames, filenames) in os.walk(external_data)]
    filtered_files = list(filter(lambda x: len(x[1]) > 0, directory_paths))

    output_loc = os.path.join(dirname, '../../../data/interim/onto5/')
    cased_new_file_path = os.path.join(output_loc, 'data_cased.iob2')
    uncased_new_file_path = os.path.join(output_loc, 'data_uncased.iob2')

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)

    f1 = open(cased_new_file_path, "w+")
    f2 = open(uncased_new_file_path, "w+")

    for (dirname, files) in filtered_files:
        for ner_file in files:
            old_file_path = os.path.join(dirname, ner_file)

            convert(old_file_path, cased_new_file_path, cased=True)
            convert(old_file_path, uncased_new_file_path, cased=False)


if __name__ == '__main__':
    clean_dataset()