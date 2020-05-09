import os, json, math, sys

import numpy as np
import keras.backend as K
from keras.layers import *
from keras.models import Model, model_from_json
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from keras.losses import sparse_categorical_crossentropy
from keras import activations, initializers, regularizers, constraints, optimizers

import preprocess, dataset

config = {}
expNo = str(sys.argv[1])
PARAMS = {
    "path_to_config": "./exp"+expNo+"/config.json",
    "load_saved_model": True,
    "train_model": False,
    "evaluate_model": True,
    "predict": False,
    "model_name": "weights.05.hdf5",
    "verbose": True
}

def length_penalty(y_true, y_pred):
    sent_len = K.sum(y_true, axis=-2)
    summ_len = K.sum(y_pred, axis=-2)
    x = summ_len/sent_len
    return 150*((x-config["summ_len"])**2)

def one_hot_penalty(y_true, y_pred):
    y = (y_pred-0.5)*(y_pred-0.5)
    return 1.25-5*y

def decoder_penalty(y_true, y_pred):
    return K.sum(K.sparse_categorical_crossentropy(y_true, y_pred), axis=-1)

def coverage_penalty(y_true, y_pred):
    return K.sum(y_pred, axis=1)

def mean_sqrt_abs_loss(y_true, y_pred):
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.sqrt(K.abs(y_pred - y_true)), axis=-1)


class SentSummarizer(object):
    def __init__(self, config):
        self.config = config
        self.indexer = preprocess.Encoder(config)
        self._create_model()
        self.compile()
        self.model.summary()


    def _create_model(self):
        # load the whole embedding into memory
        embedding_matrix = np.zeros((self.config["vocab_size"], self.config["word_emb_len"]))
        f = open(self.config["glove_path"],'rt',encoding='utf-8')
        vocab_count_found = 0
        for line in f:
            values = line.split()
            word = values[0]
            if word not in self.indexer.word_to_index:
                continue
            vocab_count_found += 1
            ind = self.indexer.word_to_index[word]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix[ind] = coefs
        f.close()
        print('Loaded GLoVe word vectors.')
        print("Vectors found: ", vocab_count_found, "Vocab size: ", self.config["vocab_size"])

        cur_sent = Input(shape=(None,), dtype='int16', name="cur_sent")

        # Append start symbol to cur_sent
        cur_sent_inp = Lambda(lambda x: K.concatenate([K.cast(K.ones((K.shape(x)[0], 1)) * self.indexer.delimiter, dtype='int16'), K.cast(x, 'int16')], axis=1), name='pad_input')(cur_sent)

        # Append stop symbol to cur_sent
        cur_sent_target = Lambda(lambda x: K.concatenate([K.cast(x, 'int16'), K.cast(K.ones((K.shape(x)[0], 1)) * self.indexer.delimiter, dtype='int16')], axis=1), name='pad_target')(cur_sent)

        self.word_emb_layer = Embedding(
                            output_dim=self.config["word_emb_len"],
                            input_dim=self.config["vocab_size"],
                            name="word_embedding",
                            weights=[embedding_matrix],
                        )
        cur_sent_word_emb = self.word_emb_layer(cur_sent)

        if self.config["bilstm"]:
            self.sent_encoder = Bidirectional(LSTM(
                        self.config["lstm_size"]//2,
                        name="sent_encoder",
                        return_state=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    ))
        else:
            self.sent_encoder = LSTM(
                        self.config["lstm_size"],
                        name="sent_encoder",
                        return_state=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    )
        sent_encoder_states = self.sent_encoder(cur_sent_word_emb)[1:]

        if self.config["bilstm"]:
            self.iem_decoder = Bidirectional(LSTM(
                        self.config["lstm_size"]//2,
                        name="iem_decoder",
                        return_sequences=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    ))
        else:
            self.iem_decoder = LSTM(
                        self.config["lstm_size"],
                        name="iem_decoder",
                        return_sequences=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    )
        iem_decoder_states = self.iem_decoder(cur_sent_word_emb, initial_state=sent_encoder_states)

        self.iem_hidden = Dense(
                            self.config["iem_hidden"],
                            kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                            activation="relu",
                            name="iem_hidden",
		                )
        iem_hidden_out = self.iem_hidden(iem_decoder_states)
        self.iem_summary_mask = Dense(1,
                    kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                    activation="sigmoid",
                    name="iem_summary_mask"
                   )
        summary_mask = self.iem_summary_mask(iem_hidden_out)
        one_hot = Lambda(lambda x: K.identity(x), name="one_hot")(summary_mask)
        summary_words = multiply([cur_sent_word_emb, summary_mask])

        # is_stop predictor
        if self.config["bilstm"]:
            self.is_stop_decoder = Bidirectional(LSTM(
                        self.config["lstm_size"]//2,
                        name="is_stop_decoder",
                        return_sequences=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    ))
        else:
            self.is_stop_decoder = LSTM(
                        self.config["lstm_size"],
                        name="is_stop_decoder",
                        return_sequences=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    )
        is_stop_decoder_states = self.is_stop_decoder(cur_sent_word_emb, initial_state=sent_encoder_states)
        self.is_stop_hidden = Dense(
                            self.config["is_stop_hidden"],
                            kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                            activation="relu",
                            name="is_stop_hidden",
		                )
        is_stop_hidden_out = self.is_stop_hidden(is_stop_decoder_states)
        self.is_stop = Dense(1,
                    kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                    activation="sigmoid",
                    name="is_stop"
                   )
        is_stop = self.is_stop(is_stop_hidden_out)
        is_stop_flat = Reshape((-1,), name='is_stop_flat')(is_stop)
        is_stop_padded = Lambda(
            lambda x: K.concatenate([K.ones((K.shape(x)[0], 1, 1), dtype='float32'), x], axis=1)
        )(is_stop)
        is_content_padded = Lambda(lambda x: 1 - x, name='is_content')(is_stop_padded)

        if self.config["bilstm"]:
            self.summ_encoder = Bidirectional(LSTM(
                        self.config["lstm_size"]//2,
                        name="summ_encoder",
                        return_state=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    ))
        else:
            self.summ_encoder = LSTM(
                        self.config["lstm_size"],
                        name="summ_encoder",
                        return_state=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    )
        summ_encoder_states = self.summ_encoder(summary_words)[1:]

        summary_mask_padded = Lambda(
            lambda x: K.concatenate([K.ones((K.shape(x)[0], 1, 1), dtype='float32'), x], axis=1)
            )(summary_mask)
        cur_sent_inp_emb = self.word_emb_layer(cur_sent_inp)
        summary_words_padded = multiply([cur_sent_inp_emb, summary_mask_padded])

        if self.config["bilstm"]:
            self.summ_decoder = Bidirectional(LSTM(
                        self.config["lstm_size"]//2,
                        name="summ_decoder",
                        return_sequences=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    ))
        else:
            self.summ_decoder = LSTM(
                        self.config["lstm_size"],
                        name="summ_decoder",
                        return_sequences=True,
                        kernel_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        recurrent_initializer=initializers.RandomNormal(stddev=self.config["std"]),
                        kernel_regularizer=regularizers.l2(self.config["l2reg"]),
                        # recurrent_regularizer=regularizers.l2(self.config["l2reg"]),
                        # activity_regularizer=regularizers.l2(self.config["l2reg"])
                    )
        summ_decoder_states = self.summ_decoder(summary_words_padded, initial_state=summ_encoder_states)

        self.decoder_word_emb_layer = Dense(
            self.config["vocab_size"],
            activation="softmax",
            kernel_initializer=initializers.RandomUniform(minval=-1*self.config["std"], maxval=self.config["std"]),
            use_bias=False
        )
        summ_decoder_out = self.decoder_word_emb_layer(summ_decoder_states)
        summ_decoder_out = Lambda(lambda x: K.identity(x), name="summ_decoder_out")(summ_decoder_out)
        is_stop_repeat = Lambda(
            lambda x: K.repeat_elements(
                x,
                self.config["stop_list_size"] + 1,
                -1
                ))(is_stop_padded)
        is_content_repeat = Lambda(
            lambda x: K.repeat_elements(
                x,
                self.config["vocab_size"] - self.config["stop_list_size"] - 1,
                -1
                ))(is_content_padded)
        re_weight = Concatenate(axis=-1)([is_stop_repeat, is_content_repeat])
        summ_decoder_out_re_weighted = Multiply()([summ_decoder_out, re_weight])
        summ_decoder_out_re_weighted = Softmax(axis=-1)(summ_decoder_out_re_weighted)

        # Loss linking summary and decoder
        word_loss = Lambda(lambda x: K.sparse_categorical_crossentropy(x[0], x[1]))([cur_sent_target, summ_decoder_out_re_weighted])
        word_loss_norm = Lambda(lambda x: x/K.max(x, axis=1, keepdims=True))(word_loss)
        summary_mask_target = Lambda(
            lambda x: K.concatenate([x, K.zeros((K.shape(x)[0], 1, 1), dtype='float32')], axis=1)
            )(summary_mask)
        summary_mask_flat = Reshape((-1,))(summary_mask_target)
        coverage = Lambda(lambda x: x[0]*K.exp(1-x[1]) + (1-x[0])*K.exp(x[1]) - 1, name="coverage")([summary_mask_flat, word_loss_norm])
        # coverage = Lambda(lambda x: x[0]*(1-x[1]) + (1-x[0])*x[1], name="coverage")([summary_mask_flat, word_loss_norm])

        self.model = Model(
            [cur_sent],
            [summary_mask, one_hot, summ_decoder_out, coverage, is_stop_flat]
            )


    def compile(self):
        # optimizer = optimizers.Adam(clipnorm=self.config["gclip"], clipvalue=0.5)
        optimizer = optimizers.RMSprop(clipnorm=self.config["gclip"], clipvalue=0.5)
        summ_length_loss = self.config["summ_length_loss"]
        binarization_loss = self.config["binarization_loss"]
        summ_reconstruction_loss = self.config["summ_reconstruction_loss"]
        linkage_loss = self.config["linkage_loss"]
        is_stop_loss = self.config["is_stop_loss"]

        self.model.compile(
            optimizer=optimizer,
            loss={
                "iem_summary_mask": length_penalty,
                "one_hot": one_hot_penalty,
                "summ_decoder_out": decoder_penalty,
                "coverage": coverage_penalty,
                "is_stop_flat": 'mse',
                },
            loss_weights={
                "iem_summary_mask": summ_length_loss,
                "one_hot": binarization_loss,
                "summ_decoder_out": summ_reconstruction_loss,
                "coverage": linkage_loss,
                "is_stop_flat": is_stop_loss,
                },
            )


    def train(self):
        csv_logger = CSVLogger(
            os.path.join(self.config["save_path"], self.config["logfile"]),
            append=True,
            separator=',')
        modelCheckpoint = ModelCheckpoint(
            os.path.join(self.config["save_path"], self.config["weights_path"]),
            # monitor="val_loss",
            # save_best_only=True,
            save_weights_only=True
        )
        train_data_gen = dataset.batch_generator(self.config, "train")
        valid_data_gen = dataset.batch_generator(self.config, "valid")
        self.model.fit_generator (
            train_data_gen,
            steps_per_epoch=self.config["no_of_steps"],
            epochs=self.config["epochs"],
            validation_data=valid_data_gen,
            validation_steps=self.config["no_of_steps_valid"],
            callbacks=[csv_logger, modelCheckpoint],
            shuffle=False
        )


    def evaluate(self):
        valid_gen = dataset.batch_generator(self.config, "test")
        for inp in valid_gen:
            prediction = self.model.predict_on_batch(inp[0])
            summary_mask, one_hot, decoder, coverage, is_stop = prediction
            for sent_no, sent in enumerate(inp[0][0]):
                words = []
                summ = []
                pred = []
                dec_losses = []
                dec_loss_total = 0
                dec_losses_pred = []
                keep_count = 0
                for word_no, word_id in enumerate(sent):
                    word = str(self.indexer.ind_2_word[word_id])
                    words.append(word)

                    pred_word_id = np.argmax(decoder[sent_no][word_no])
                    pred_word = str(self.indexer.ind_2_word[pred_word_id])
                    pred.append(pred_word)
                    dec_loss_pred = -1 * math.log(decoder[sent_no][word_no][pred_word_id])
                    dec_losses_pred.append(round(dec_loss_pred, 2))

                    dec_loss = -1 * math.log(decoder[sent_no][word_no][word_id])
                    dec_loss_total += dec_loss
                    dec_losses.append(round(dec_loss, 2))

                    if summary_mask[sent_no][word_no] >= 0.5:
                        summ.append(word)
                        keep_count += 1
                    else:
                        summ.append("-" * len(word))

                print("Sent: ", ' '.join(words))
                print("Summ: ", ' '.join(summ))
                if PARAMS["verbose"]:
                    print("Pred: ", ' '.join(pred))
                    print("Is Stop: ", [round(each, 2) for each in is_stop[sent_no]])
                    print("Sent len: ", len(words))
                    print("Summ ratio: ", round(keep_count / len(words), 2))
                    print("Decoder Pred: ", dec_losses_pred)
                    print("Decoder Orig: ", dec_losses)
                    print("Linkage: ", [round(each, 2) for each in coverage[sent_no]])
                    print("Decoder Loss: ", round(dec_loss_total, 2))
                input(">>")


    def predict(self, sents):
        encodings = []
        for sent in sents:
            encoding, _ = self.indexer.encode_sent(sent)
            encodings.append(encoding)
        return self.model.predict_on_batch(np.asarray(encodings, 'int16'))


def create_batch(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    global config
    config = json.loads(open(PARAMS["path_to_config"]).read())
    model = SentSummarizer(config)
    if PARAMS["load_saved_model"]:
        model.model.load_weights(os.path.join(config["save_path"], PARAMS["model_name"]))
    if PARAMS["train_model"]:
        model.train()
    if PARAMS["evaluate_model"]:
        model.evaluate()
    if PARAMS["predict"]:
        predictFile = str(sys.argv[2])
        f = open(predictFile)
        sents_by_len = {}
        for line_no, line in enumerate(f):
            sent = preprocess.filter_words(line.rstrip('\n').split())
            if len(sent) not in sents_by_len:
                sents_by_len[len(sent)] = []
            sents_by_len[len(sent)].append((line_no, ' '.join(sent)))
        f.close()
        summaries = []
        for slen, sents in sents_by_len.items():
            for batch_with_line_no in create_batch(sents, 128):
                line_nos = [each[0] for each in batch_with_line_no]
                batch = [each[1] for each in batch_with_line_no]
                summary_masks, _, decoder, _, _ = model.predict(batch)
                for sent_no, mask in enumerate(summary_masks):
                    summary = []
                    pred = []
                    dec_loss_total = 0
                    sent_words = batch[sent_no].split()
                    for word_no, word_mask in enumerate(mask):
                        word = sent_words[word_no]
                        word_id = -1
                        if word[0] == "#":
                            word_id = model.indexer.word_to_index[model.indexer.num_char]
                        elif word in model.indexer.word_to_index:
                            word_id = model.indexer.word_to_index[word]
                        else:
                            word_id = model.indexer.word_to_index[model.indexer.oov_char]

                        pred_word_id = np.argmax(decoder[sent_no][word_no])
                        pred_word = str(model.indexer.ind_2_word[pred_word_id])
                        pred.append(pred_word)
                        dec_loss = -1 * math.log(decoder[sent_no][word_no][word_id])
                        dec_loss_total += dec_loss

                        if word_mask[0] >= 0.5:
                            summary.append(word)
                    dec_loss_total /= len(mask)
                    summaries.append(
                        #(dec_loss_total,
                        (line_nos[sent_no],
                        ' '.join(summary),
                        dec_loss_total,
                        ' '.join(pred),
                         batch[sent_no])
                    )
        # Output summaries in same order as input
        summaries.sort()
        for summary in summaries:
            print (summary[1])
            if PARAMS['verbose']:
                print ("Orig:", summary[-1])
                print ("Reconstruction loss:", summary[2])
                print ("Reconstructed sent:", summary[3])
                print ("Summ factor:", round(len(summary[1].split())/len(summary[-1].split()),2))


if __name__ == "__main__":
    main()
