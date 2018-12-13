import os
from keras import backend
from keras.layers import *
from keras.models import Model
import json
import shlex
import subprocess
from keras import regularizers
from keras import optimizers
import random
import datetime
import re
import math
import sys

rels = json.load(open("qrels/rels08.json"))


def make_train_data(qid, img_direc='img'):
    if qid not in rels:
        return None, None

    pos_list, neg_list = [], []

    pos_pool = [doc for rating in ['2', '1'] if rating in rels[qid] for doc in rels[qid][rating]]
    neg_pool = [doc for rating in ['0'] if rating in rels[qid] for doc in rels[qid][rating]]

    pl, nl = len(pos_pool), len(neg_pool)

    if pl == 0 or nl == 0:
        return None, None

    pos_docs = pos_pool
    neg_docs = random.sample(neg_pool * int(math.ceil(pl / nl)), pl)

    random.shuffle(pos_docs)
    random.shuffle(neg_docs)

    for pos_doc, neg_doc in zip(pos_docs, neg_docs):
        pos_mat = np.load(str(os.path.join(img_direc, qid, pos_doc)) + ".npy")[:, :, :]

        neg_mat = np.load(str(os.path.join(img_direc, qid, neg_doc)) + ".npy")[:, :, :]

        pos_list.append(pos_mat)
        neg_list.append(neg_mat)

    return np.asarray(pos_list), np.asarray(neg_list)


def make_test_data(fold, img_direc="img_tiling"):
    test_set_map = {
        1: "5",
        2: "1",
        3: "2",
        4: "3",
        5: "4"
    }
    doc_mat = []
    for line in open("qrels/MQ2008" + "_S" + test_set_map[fold] + ".txt"):
        parts = line.split()
        qid, doc = parts[0], parts[2]
        mat = np.load(str(os.path.join(img_direc, qid, doc)) + ".npy")[:, :, :]
        doc_mat.append(mat)
    return np.asarray(doc_mat)


def train(pos_docs, neg_docs, epochs=10):
    query_len, seg_len = 9, 30
    input_shape = (query_len, seg_len, 2)

    kernel_widths = [i for i in range(1, 11)]

    convs = [Conv2D(3, (query_len, width), kernel_regularizer=regularizers.l2()) for width in kernel_widths]

    reshapes = [Reshape((seg_len - width + 1, 3)) for width in kernel_widths]

    lstms = [LSTM(3) for width in kernel_widths]

    d1 = Dense(32, activation='relu', kernel_regularizer=regularizers.l2())

    d2 = Dense(16, activation='relu', kernel_regularizer=regularizers.l2())
    s = Dense(1)

    pos_input = Input(input_shape)
    neg_input = Input(input_shape)

    pos_convs = [reshape(conv(pos_input)) for reshape, conv in zip(reshapes, convs)]

    pos_lstms = [lstm(pos_conv) for lstm, pos_conv in zip(lstms, pos_convs)]

    pos_concate = concatenate(pos_lstms)

    pos_d1 = d1(pos_concate)
    pos_d2 = d2(pos_d1)
    pos_score = s(pos_d2)

    neg_convs = [reshape(conv(neg_input)) for reshape, conv in zip(reshapes, convs)]

    neg_lstms = [lstm(neg_conv) for lstm, neg_conv in zip(lstms, neg_convs)]

    neg_concate = concatenate(neg_lstms)

    neg_d1 = d1(neg_concate)
    neg_d2 = d2(neg_d1)
    neg_score = s(neg_d2)

    negate_neg_score = Lambda(lambda x: -1 * x, output_shape=(1,))(neg_score)

    diff = Add()([pos_score, negate_neg_score])

    prob = Activation("sigmoid")(diff)

    model = Model(inputs=[pos_input, neg_input], outputs=prob)

    optimizer = optimizers.Adam(lr=0.005)

    model.compile(optimizer=optimizer, loss="binary_crossentropy")

    y = np.ones((pos_docs.shape[0], 1))

    model.fit([pos_docs, neg_docs], y, epochs=epochs, verbose=2, batch_size=32)

    return backend.function([pos_input], [pos_score])


def evaluate(run_file):
    # run_file = re.escape(run_file)
    runs = re.escape(run_file + ".txt")
    output = re.escape(run_file + ".out")
    arg_str = "perl eval/Eval-Score-4.0.pl eval/MQ2008" + "_test.txt  result/" + runs + " result/" + output + " 0"
    args = shlex.split(arg_str)
    result = subprocess.run(args, stdout=subprocess.PIPE)
    print(result.stdout.decode())
    print(open("result/"+output).read())


def k_fold(img_direc, epochs=5):
    os.makedirs("result", exist_ok=True)
    run_file = "k_fold_" + str(datetime.datetime.now())
    for fold in range(1, 6):
        print(fold)
        # test_topics = folds[fold]
        train_topics = json.load(open("qrels/MQ2008" + "_train_" + str(fold) + ".json"))
        print(len(train_topics))
        random.shuffle(train_topics)

        pos_data, neg_data = None, None
        for i in train_topics:
            # print(i)
            pos, neg = make_train_data(str(i), img_direc)
            if pos is None:
                # print(i)
                continue
            if pos_data is None:
                pos_data = pos
                neg_data = neg
            else:
                pos_data = np.vstack((pos_data, pos))
                neg_data = np.vstack((neg_data, neg))

        print(pos_data.shape, neg_data.shape)
        scorer = train(pos_data, neg_data, epochs=epochs)

        doc_mat = make_test_data(fold, img_direc)
        scores = scorer([doc_mat])[0]

        with open(os.path.join("result", run_file + ".txt"), "a") as output:
            output.write(
                "\n".join([str(scores[i, 0]) for i in range(doc_mat.shape[0])])
                + "\n"
            )

    evaluate(run_file)


if __name__ == "__main__":
    k_fold(sys.argv[1], int(sys.argv[2]))
