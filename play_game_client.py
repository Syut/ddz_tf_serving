from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

import time
import random
import numpy as np
from tensorflow import make_tensor_proto

from input_trans import play_game_input
from trans_utils import str2ary, ary2str
from split_cards import kicker_append
from hand_type import HAND_LABEL2CHAR, HAND_CHAR2LABEL, str2label
from kicker_type import *
from kicker_input_trans import build_kicker_input


def get_stub(hostport):
    host, port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    return prediction_service_pb2.beta_create_PredictionService_stub(channel)


def hand_req(img, lb):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ddz'
    request.model_spec.signature_name = 'predict_hand'
    image = img.astype(np.float32)
    label = lb.astype(np.float32)
    request.inputs['images'].CopyFrom(make_tensor_proto(image, shape=[1, 21, 19, 15]))
    request.inputs['legal'].CopyFrom(make_tensor_proto(label, shape=[1, 309]))
    return request


def kicker_req(img):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'ddz'
    request.model_spec.signature_name = 'predict_kicker'
    image = img.astype(np.float32)
    request.inputs['images'].CopyFrom(make_tensor_proto(image, shape=[1, 3, 9, 15]))
    return request


def append_kicker(cards, pot, process, role, hand, stub):
    main_hand = str2ary(hand[:-1])
    cur_type = hand[-1]
    kicker_len, kicker_width = KICKER_PARAMS[cur_type]
    kicker_type = KICKER_TYPE[cur_type]
    ret_kicker = np.zeros(15, dtype=np.int32)
    remain = np.copy(cards)
    remain -= main_hand
    recorder = np.copy(pot) if role == 0 else np.zeros(15, dtype=np.int32)
    for p in process:
        cur_role, cur_hand = p
        hand_pot = np.copy(cur_hand)
        if cur_role == 0 and np.sum(pot) > 0:
            hand_pot -= pot
            num = np.where(hand_pot < 0)[0]
            pot = np.zeros(15, dtype=np.int32)
            for k in num:
                pot[k] = -hand_pot[k]
                hand_pot[k] += pot[k]
        if cur_role == role:
            remain -= cur_hand
        recorder = recorder + hand_pot if role == 0 else recorder + cur_hand
    cur_mains = []
    cur_mains_index = np.where(main_hand == np.max(main_hand))[0]
    for i in cur_mains_index:
        cur_main = np.zeros(15, dtype=np.int32)
        cur_main[i] = 1
        cur_mains.append(cur_main)
    while len(cur_mains) < kicker_len:
        cur_mains.append(cur_main)
    for cur_main in cur_mains:
        x_input_k = build_kicker_input(kicker_type, role, main_hand, remain, kicker_width, kicker_len, cur_main, recorder,
                                       ret_kicker)
        request = kicker_req(x_input_k)
        result_future = stub.Predict.future(request, 1)
        kicker = np.array(result_future.result().outputs['labels'].int_val)[0]
        for j in range(kicker_width):
            ret_kicker[kicker] += 1
    kicker_str = ary2str(ret_kicker)
    check_legal = np.copy(cards)
    for p in process:
        if p[0] == role:
            check_legal -= p[1]
    check_legal -= main_hand
    check_legal -= ret_kicker
    check_mask = check_legal < 0
    temp_out = hand[:-1] + kicker_str
    if True in check_mask or isinstance(str2label(temp_out), str):
        check_legal += ret_kicker
        out_hand = kicker_append(check_legal, HAND_CHAR2LABEL[hand])
    else:
        out_hand = temp_out
    return out_hand


def get_hand(top_n_label, probs, score, random_play):
    if not random_play:
        return top_n_label[0], probs[0]
    else:
        rd = random.random()
        out_hand = -1
        prob = 0
        for l, p in (top_n_label, probs):
            if rd < p and out_hand < 0:
                out_hand = l
                prob = p
                break
            else:
                rd -= p
        if out_hand == -1:
            out_hand = top_n_label[0]
            prob = probs[0]
        else:
            if score[out_hand] == 1:
                out_hand = top_n_label[0]
                prob = probs[0]
        return out_hand, prob


def get_one_hand(cards, process, role, pot, random_play=False, hostport='localhost:9000'):
    stub = get_stub(hostport)
    image, label = play_game_input(cards, process, role)
    request = hand_req(image, label)
    result_future = stub.Predict.future(request, 1)
    top_n_labels = np.array(result_future.result().outputs['labels'].int_val)
    probs = np.array(result_future.result().outputs['scores'].float_val)
    out_hand_type, prob = get_hand(top_n_labels, probs, label, random_play)
    if 130 <= out_hand_type <= 223 or 269 <= out_hand_type <= 294:
        hand = HAND_LABEL2CHAR[out_hand_type]
        out_hand = append_kicker(cards, pot, process, role, hand, stub)
    else:
        out_hand = HAND_LABEL2CHAR[out_hand_type]
    return out_hand, prob


def get_top_n_hand(cards, process, role, pot, hostport='localhost:9000'):
    stub = get_stub(hostport)
    image, label = play_game_input(cards, process, role)
    request = hand_req(image, label)
    result_future = stub.Predict.future(request, 1)
    out_hand_type = np.array(result_future.result().outputs['labels'].int_val)
    probs = np.array(result_future.result().outputs['scores'].float_val)
    out_hands = []
    out_probs = []
    for i in range(len(out_hand_type)):
        if probs[i] > 0:
            if 130 <= out_hand_type[i] <= 223 or 269 <= out_hand_type[i] <= 294:
                hand = HAND_LABEL2CHAR[out_hand_type[i]]
                out_hand = append_kicker(cards, pot, process, role, hand, stub)
            else:
                out_hand = HAND_LABEL2CHAR[out_hand_type[i]]
            out_hands.append(out_hand)
            out_probs.append(probs[i])
    return out_hands, out_probs


def get_game_result(game_ary, pot, turn=None, process=None, random_play=False, hostport='localhost:9000'):
    stub = get_stub(hostport)
    out_hands = [] if process is None else process.copy()
    role = 0 if turn is None else turn
    cur_cards = np.copy(game_ary)
    for p in out_hands:
        cur_cards[p[0]] -= p[1]
    while True:
        image, label = play_game_input(game_ary[role], out_hands, role)
        request = hand_req(image, label)
        result_future = stub.Predict.future(request, 1)
        top_n_labels = np.array(result_future.result().outputs['labels'].int_val)
        probs = np.array(result_future.result().outputs['scores'].float_val)
        out_hand_type, prob = get_hand(top_n_labels, probs, label, random_play)
        if 130 <= out_hand_type <= 223 or 269 <= out_hand_type <= 294:
            hand = HAND_LABEL2CHAR[out_hand_type]
            out_hand = append_kicker(game_ary[role], pot, out_hands, role, hand, stub)
        else:
            out_hand = HAND_LABEL2CHAR[out_hand_type]

        out_hand_ary = str2ary(out_hand)
        cur_cards[role] -= out_hand_ary
        out_hands.append((role, out_hand_ary))
        if np.sum(cur_cards[role]) == 0:
            break
        elif np.sum(cur_cards[role]) < 0:
            print('-----error-------')
            print(cur_cards)
            temp = []
            for i in game_ary:
                temp.append(ary2str(i))
            print(';'.join(temp))
            temp = []
            for i in process:
                temp.append(','.join((str(i[0]), ary2str(i[1]))))
            print(';'.join(temp))
            return
        else:
            role += 1
            role = role % 3
    return role, out_hands


if __name__ == '__main__':
    hostport = "192.168.31.196:9000"
    g = '333678899JQQKA22D;34445667TTTTJJQA2;4555677889JQKAA2X;9KK'
    pg = '0,3336;1,P'
    prc = pg.split(';')
    rounds_ary = []
    for i in prc:
        cur_role, hand = i.split(',')
        rounds_ary.append((int(cur_role), str2ary(hand)))
    game_ary = str2ary(g, separator=';')
    t1 = time.time()
    hand = get_one_hand(game_ary[2], rounds_ary, 2, game_ary[3], hostport=hostport)
    t2 = time.time()
    print(hand)
    print(t2 - t1)
    game_ary[0] += game_ary[3]
    w, pc = get_game_result(game_ary, game_ary[3], hostport=hostport)
    temp = []
    for i in pc:
        temp.append(','.join((str(i[0]), ary2str(i[1]))))
    pc_str = ';'.join(temp)
    print(w, pc_str)
