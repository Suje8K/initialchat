# encoding: UTF-8
#!flask/bin/python
#import tensorflow as tf
from tensorflow import Session, train
import numpy as np
import my_txtutils
from flask import Flask
from flask import request
import json

app = Flask(__name__)

# these must match what was saved !
ALPHASIZE = my_txtutils.ALPHASIZE
NLAYERS = 3
INTERNALSIZE = 512

A = 'Sujeet Kumar Roy Polaris';
B = 'Neha Jaiswal';

pythonB10 = "checkpoints/rnn_train_1506774504-12000000"  # can even recite the Apache license
author = pythonB10

@app.route('/')
def processingContext():
    queryText = request.args.get('txt')
    if(str(queryText).startswith('A')):
	    queryText = A + queryText[1:]
    else:
        queryText = B + queryText[1:]
    respAmt = int(request.args.get('rspAmt'))
    prm = request.args.get('json')
    queryText = queryText + "*"
    with Session() as sess:
        new_saver = train.import_meta_graph('checkpoints/rnn_train_1506774504-12000000.meta')
        new_saver.restore(sess, author)
        x = my_txtutils.convert_from_alphabet(ord(queryText[0]))
        x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

        # initial values
        y = x
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
        print(queryText[:len(queryText)-1])
        for jj in range(len(list(queryText))):
            yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
            if(queryText[jj+1] == '*'):
                break
            c = my_txtutils.convert_from_alphabet(ord(queryText[jj+1]))
            y = np.array([[c]])
        if(str(prm) == 'json'):
            return json.dumps({"responses":processingResponses(h, y, respAmt, sess, queryText[0])})
        elif(str(prm) == 'jsonPP'):
            return json.dumps({"responses":processingResponses(h, y, respAmt, sess, queryText[0])}, indent=4)
        else:
            return '<br>'.join(processingResponses(h, y, respAmt, sess, queryText[0]))

def processingResponses(h, y, respAmt, sess, userFirstChar):
    rsps = []
    for iii in range(respAmt):
        ignoreResp, hh, yy = computeNextSeq(h, y, "*", sess, userFirstChar)
        currResp, _, _ = computeNextSeq(hh, yy, "-", sess, userFirstChar)
        #print("".join(ignoreResp))
        tmp = "".join(currResp)
        tst = ''
        if(tmp.startswith('S')):
            tst = 'A' + tmp[len(A):]
        else:
            tst = 'A' + tmp[len(A):]
        rsps.append(tst)
    return rsps

def computeNextSeq(h, y, c, sess, userFirstChar):
    resp = []
    cc = c
    hh =h
    yy=y
    while(c != '\n'):
        yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1., 'Hin:0': h, 'batchsize:0': 1})
        c = my_txtutils.sample_from_probabilities(yo, topn=2)
        y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
        c = chr(my_txtutils.convert_to_alphabet(c))
        if(cc == '-' and c == userFirstChar):
            #print(cc+c,end=cc)
            h = hh
            y = yy
            continue
        #print(c,end="")
        if c == '\n':
            break
        resp.append(c)
    return resp, h, y


if __name__ == '__main__':
    #processingContext(user2 + action, 10)
    app.run(debug=True)
