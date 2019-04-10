import os
import sys
import re

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


def check_and_process_eos(out, cur_type, token, prefix=''):
    match = re.match(EOS_PATTERN, token)
    if match:
        out.write(match.group(1) + '\t' + prefix + cur_type + '\n')
        out.write('.' + '\t' + cur_type + '\n')
        out.write('\n')
        return True
    return False


def convert(file_from, file_to):
    cur_type = NON_ENTITY_TYPE

    with open(file_from, 'r') as f, open(file_to, 'w') as out:
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
                        out.write(match.group(1) + '\t' + NON_ENTITY_TYPE + '\n')
                    continue

                match = re.match(END_SINGLE_PATTERN, token)
                if match:
                    out.write(match.group(2) + '\t' + 'B-' + match.group(1) + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(out, cur_type, match.group(3)) and match.group(3):
                        out.write(match.group(3) + '\t' + cur_type + '\n')
                    continue

                match = re.match(END_SINGLE_OTHER_ATTR_PATTERN, token)
                if match:
                    out.write(match.group(2) + '\t' + 'B-' + cur_type + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(out, cur_type, match.group(3)) and match.group(3):
                        out.write(match.group(3) + '\t' + cur_type + '\n')
                    continue

                match = re.match(TYPE_PATTERN, token)
                if match:
                    cur_type = match.group(1)
                    out.write(match.group(2) + '\t' + 'B-' + cur_type + '\n')
                    continue

                match = re.match(TYPE_NO_END_PATTERN, token)
                if match:
                    cur_type = match.group(1)
                    continue
                
                match = re.match(OTHER_ATTR_PATTERN, token)
                if match:
                    cur_type = match.group(1)
                    out.write(match.group(2) + '\t' + 'B-' + cur_type + '\n')
                    continue

                match = re.match(END_MULTI_PATTERN, token)
                if match:
                    out.write(match.group(1) + '\t' + 'I-' + cur_type + '\n')
                    cur_type = NON_ENTITY_TYPE
                    if not check_and_process_eos(out, cur_type, match.group(2)) and match.group(2):
                        out.write(match.group(2) + '\t' + cur_type + '\n')
                    continue

                if check_and_process_eos(out, cur_type, token, 'I-'):
                    continue

                if len(token) == 2 and token[0] == '/':
                    token = token[1]

                if cur_type != NON_ENTITY_TYPE:
                    out.write(token + '\t' + 'I-' + cur_type + '\n')
                else:
                    out.write(token + '\t' + cur_type + '\n')

            out.write('\n')

def clean_dataset():
    print("Cleaning dataset...")
    dirname = os.path.dirname(__file__)  # NOQA: E402
    external_data = os.path.join(dirname, '../../data/external/onto5/english/annotations/')
    directory_paths = [(dirpath, list(filter(lambda x: x.endswith('.name'), filenames))) for (dirpath, dirnames, filenames) in os.walk(external_data)]
    filtered_files = list(filter(lambda x: len(x[1]) > 0, directory_paths))

    output_loc = os.path.join(dirname, '../../data/interim/onto5/english/annotations/')

    if not os.path.exists(output_loc):
        os.makedirs(output_loc)

    for (dirname, files) in filtered_files:
        for ner_file in files:
            abs_path = os.path.abspath(dirname)
            sub_path = abs_path.split('onto5/english/annotations/', 1)[1]
            new_dir = os.path.join(output_loc, sub_path)
            
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            old_file_path = os.path.join(dirname, ner_file)
            new_file_path = os.path.join(new_dir, ner_file)

            convert(old_file_path, new_file_path)

if __name__ == '__main__':
    clean_dataset()