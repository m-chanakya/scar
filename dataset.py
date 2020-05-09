import os, json
import random
import numpy as np
import preprocess


def batch_generator(config, mode="train", seed=1267):
    batch_files = {}
    is_stop_files = {}
    is_batch_file_empty = {}
    indexer = preprocess.Encoder(config)
    batch_start_symbol = np.repeat([[indexer.delimiter]], config["batch_size"], axis=0)

    if mode=="train":
      batch_path = config["path_to_data"]
      is_stop_path = config["path_to_is_stop"]
    else:
      batch_path = config["path_to_validation_data"]
      is_stop_path = config["path_to_validation_is_stop"]

    slens = range(config["min_sent_len"], config["max_sent_len"]+1)

    for slen in slens:
        batch_file = batch_path.format(slen)
        batch_files[slen] = open(batch_file, 'rb')
        is_stop_file = is_stop_path.format(slen)
        is_stop_files[slen] = open(is_stop_file, 'rb')
        is_batch_file_empty[slen] = False #Flag to indicate file is empty

    random.seed(seed)
    while True:
        #Choose random len file
        while True:
            slen = random.choice(slens)
            if not is_batch_file_empty[slen]:
                break
        batch_file = batch_files[slen]
        is_stop_file = is_stop_files[slen]

        cur_batch = np.zeros((config["batch_size"], slen), dtype='int16')
        is_stop_batch = np.zeros((config["batch_size"], slen), dtype='int16')

        sent_no = 0
        while sent_no < config["batch_size"]:
            sent = np.fromfile(batch_file, 'int16', slen)
            is_stop = np.fromfile(is_stop_file, 'int16', slen)
            if sent.size == 0: # Finished reading file
                batch_file.seek(0) # Rewind file
                is_stop_file.seek(0)
                is_batch_file_empty[slen] = True

                #Epoch is over if all files are empty
                end_epoch = True
                for each in slens:
                    if not is_batch_file_empty[each]:
                        end_epoch = False
                        break
                #Restart epoch
                if end_epoch:
                    for each in slens:
                        is_batch_file_empty[each] = False
                continue
            cur_batch[sent_no] = sent
            is_stop_batch[sent_no] = is_stop
            sent_no += 1

        # Add start symbol
        # cur_inp_batch = np.concatenate((batch_start_symbol, cur_batch), axis=-1)

        # Add end symbol
        cur_out_batch = np.concatenate((cur_batch, batch_start_symbol), axis=-1)

        cur_batch_len = np.ones((config["batch_size"], slen, 1))
        yield (
            [cur_batch],
            [cur_batch_len,
             cur_batch_len,
             np.asarray(np.expand_dims(cur_out_batch, -1), 'int16'),
             np.ones((config["batch_size"], slen)),
             is_stop_batch]
        )

if __name__ == "__main__":
    config = json.loads(open("./exp7/config.json", 'r').read())
    gen = batch_generator(config, config["path_to_validation_data"])
    inp, out = gen.__next__()
    sent = inp
    slen, slen, out, coverage, is_stop = out
    print("Input shapes: ")
    print("Sent: ", sent[0].shape)
    print("Output shapes: ")
    print("Summary len: ", slen.shape)
    print("One hot: ", slen.shape)
    print("Summary decoder: ", out.shape)
    print("Coverage: ", coverage.shape)
    print("Is stop: ", is_stop.shape)

