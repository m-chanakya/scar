import os, json, math
import numpy as np

class Encoder:
    def __init__(self, config):
        self.num_words = config["vocab_size"]
        index = 0
        self.word_to_index = {}

        # Index stops
        self.stops = ['the', 'a', 'of', 'to', 'in', 'and', 'on', "'s", 'for', 'that', 'with', 'at', 'an', 'as', 'from', 'by', 'has', 'his', 'its', 'was', 'after', 'is', 'will', 'it', 'their', 'have', 'over', 'be', 'were', 'he', 'here', 'first', 'who', 'against', 'this', 'more', 'up', 'reported', 'us', 'are', 'into', 'but', 'they', 'than', 'about']
        for word in self.stops:
            self.word_to_index[word] = index
            index += 1

        # Index special characters
        self.delimiter = index
        self.oov_char = index+1
        self.num_char = index+2
        self.special_sym = [self.delimiter, self.oov_char, self.num_char]
        for sym in self.special_sym:
            self.word_to_index[sym] = index
            index += 1

        # Index vocab
        vocab_file = open(config["path_to_vocab"], 'r')
        for line in vocab_file:
            if index == self.num_words:
                break
            word = line.split(':')[0]
            if word in self.stops:
                # Already indexed
                continue
            self.word_to_index[word] = index
            index += 1

        self.ind_2_word = {v:k for k,v in self.word_to_index.items()}


    def encode_sent(self, sent):
        '''
        Indexes each word in sent
        '''
        words = sent.split()
        encoding = []
        is_stop = []
        for word in words:
            if word[0] == "#":
                encoding.append(self.num_char)
                is_stop.append(0)
            elif word in self.word_to_index:
                encoding.append(self.word_to_index[word])
                if word in self.stops:
                    is_stop.append(1)
                else:
                    is_stop.append(0)
            else:
                encoding.append(self.word_to_index[self.oov_char])
                is_stop.append(0)
        return (encoding, is_stop)

    def encode_file(self, fname):
        encodings = []
        is_stops = []
        with open(fname, 'r') as f:
            for line in f:
                encoding, is_stop = self.encode_sent(line.rstrip('\n'))
                encodings.append(encoding)
                is_stops.append(is_stop)

        return (np.asarray(encodings, 'int16'), np.asarray(is_stops, 'int16'))


def filter_words(words):
    punct = [".", ",", "``", "''", "-lrb-", "-rrb-"]
    return [word for word in words if word not in punct]


def main():
    expNo = str(sys.argv[1])
    configPath = "./exp"+expNo+"/config.json"
    config = json.loads(open(configPath).read())

    #Make it configurable
    sent_lim = 37500000
    sent_counter = {} #Counts num of sents of a given len

    #Bucket data by len
    inpath= config["path_to_data_buckets"]
    inpath_valid = config["path_to_validation_data_buckets"]
    outfiles = []
    outfiles_valid = []
    for slen in range(config["min_sent_len"], config["max_sent_len"]+1):
        infile = inpath.format(slen)
        infile_valid = inpath_valid.format(slen)
        outfiles.append(open(infile, "w"))
        outfiles_valid.append(open(infile_valid, "w"))
        sent_counter[slen] = 0

    vocab = {}
    f = open(config["path_to_raw_data"], "r")
    for line in f:
        words = filter_words(line.split())
        if len(words) <  config["min_sent_len"] or len(words) > config["max_sent_len"]:
            continue
        if sent_counter[len(words)] >= sent_lim:
            continue
        for word in words:
            if word[0] == '#' or word == '<unk>':
                continue
            vocab[word] = vocab.get(word, 0) + 1
        outfiles[len(words)-config["min_sent_len"]].write(' '.join(words) + '\n')
        sent_counter[len(words)] += 1
    for each in outfiles:
        each.close()
    f.close()
    print("Created bucketed training data")

    f = open(config["path_to_raw_validation_data"], "r")
    for line in f:
        words = filter_words(line.split())
        if len(words) <  config["min_sent_len"] or len(words) > config["max_sent_len"]:
            continue
        outfiles_valid[len(words) - config["min_sent_len"]].write(' '.join(words) + '\n')
    for each in outfiles_valid:
        each.close()
    f.close()
    print("Created bucketed validation data")

    #Create vocabulary
    sorted_vocab = sorted(vocab.items(), key=lambda x:x[1], reverse=True)
    f = open(config["path_to_vocab"], "w")
    for each in sorted_vocab:
        f.write(each[0] + ":" + str(each[1]) + "\n")
    f.close()
    print("Created vocab file")

    #Index data
    outpath = config["path_to_data"]
    outpath_is_stop = config["path_to_is_stop"]
    outpath_valid = config["path_to_validation_data"]
    outpath_valid_is_stop = config["path_to_validation_is_stop"]
    encoder = Encoder(config)
    no_of_steps = 0
    no_of_steps_valid = 0
    for slen in range(config["min_sent_len"], config["max_sent_len"]+1):
        infile = inpath.format(slen)
        outfile = outpath.format(slen)
        outfile_is_stop = outpath_is_stop.format(slen)

        infile_valid = inpath_valid.format(slen)
        outfile_valid = outpath_valid.format(slen)
        outfile_valid_is_stop = outpath_valid_is_stop.format(slen)

        if os.path.exists(infile):
            encodings, is_stops = encoder.encode_file(infile)
            encodings.tofile(outfile)
            is_stops.tofile(outfile_is_stop)
            no_of_steps += math.ceil(encodings.shape[0] / float(config["batch_size"]))

        if os.path.exists(infile_valid):
            encodings, is_stops = encoder.encode_file(infile_valid)
            encodings.tofile(outfile_valid)
            is_stops.tofile(outfile_valid_is_stop)
            no_of_steps_valid += math.ceil(encodings.shape[0]/float(config["batch_size"]))

        print("Done with bucket ", slen)
    print("Finished indexing data")
    print("Training steps/epoch: ", no_of_steps)
    print("Validation steps/epoch: ", no_of_steps_valid)

if __name__ == "__main__":
    main()
