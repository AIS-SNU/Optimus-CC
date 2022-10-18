#This code attach unique key to wikipedia and wikitext to distingush two files.

import json
import multiprocessing as mp
from multiprocessing import Pool
import time
import sys
import uuid


def print_progress(prefix, start_time, num_docs,
                    num_written_docs, num_skipping_docs):

    string = prefix + ' | '
    string += 'elapsed time: {:.2f} | '.format(time.time() - start_time)
    string += 'documents: {} | '.format(num_docs)
    string += 'key-attached docs'.format(num_written_docs)
    string += 'skipping-docs: {} | '.format(num_skipping_docs)

    print(string, flush=True)


def attach_key(json_line):
    global num_docs, num_written_docs, num_skipping_docs, print_interval, start_time
     
    c_proc = mp.current_process()

    out_filename_local = output_filename.split('.')[0] + c_proc.name +"." +output_filename.split('.')[1]

    try :
        num_docs += 1

        myjson = json.loads(json_line)
        myjson['unique_key'] = input_filename.split('.')[0] + '_' + uuid.uuid4().hex
        myjson = json.dumps(myjson, ensure_ascii=False)

        with open(out_filename_local, 'ab') as f:
            f.write(myjson.encode('utf-8'))
            f.write('\n'.encode('utf-8'))
            num_written_docs += 1
            if num_docs % print_interval == 0:
                    print_progress('[PROGRESS]', start_time, num_docs,
                                    num_written_docs, num_skipping_docs)

    except Exception as e:
        num_skipping_docs += 1
        print('    skipping ', json_line, e)


def main():

    global num_docs, num_written_docs, num_skipping_docs, print_interval, start_time

    

    p = Pool(20)
    with open(input_filename, 'r') as fin:
        ret = p.map_async(attach_key, fin, 200)
        ret.get()

        p.close()
        p.join()

        print('Suceessfully called!')

    print_progress('[FINAL]', start_time, num_docs,
                    num_written_docs, num_skipping_docs)


if __name__ == '__main__':
    
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    print('attaching key to input files - {}'.format(input_filename))
    print('and will write the results to {}'.format(output_filename))

    num_docs = 0
    num_written_docs = 0
    num_skipping_docs = 0
    print_interval = 10000
    start_time = time.time()

    main()