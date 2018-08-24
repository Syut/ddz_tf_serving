#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

from ddz_type import CARDS_VALUE2CHAR, CARDS_CHAR2VALUE


def str2ary(cards_str, separator=','):
    """
    把字符串的牌型转换成数组
    输入中包含分隔符，就返回二维数组，不包含，则直接返回一个数组
    :param cards_str: 
    :param separator: 
    :return: 
    """
    ary = cards_str.split(separator) if cards_str.find(separator) > 0 else [cards_str]
    l = len(ary)
    ret = np.zeros([l, 15], dtype=np.int32)
    for i in range(l):
        for j in ary[i]:
            if j != 'P':
                ret[i][CARDS_CHAR2VALUE[j]] += 1
    ret = ret[0] if l == 1 else ret
    return ret


def ary2str(cards):
    """
    数组转字符串
    :param cards: 
    :return: 
    """
    buf = []
    for i in range(15):
        buf.extend([CARDS_VALUE2CHAR[i]] * cards[i])
    return ''.join(buf) if buf else 'P'


def ary2one_hot(ary):
    """
    数组转one_hot格式(4行)
    :param ary: 
    :return: 
    """
    ret = np.zeros([4, 15], dtype=np.int32)
    for i in range(ary.size):
        if ary[i] > 0:
            ret[ary[i] - 1][i] = 1
    return ret


def list2ary(cards):
    """
    
    :param cards: 
    :return: 
    """
    ret = np.zeros(15, dtype=np.int32)
    for i in cards:
        ret[i] += 1
    return ret
