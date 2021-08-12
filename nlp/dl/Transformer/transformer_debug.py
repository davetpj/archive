import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

from preprocess import *


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


DATA_IN_PATH = './data_in/'
DATA_OUT_PATH = './data_out'
TRAIN_INPUTS = 'train_inputs.npy'
TRAIN_OUTPUTS = 'train_outputs.npy'
TRAIN_TARGETS = 'train_targets.npy'
DATA_CONFIG = 'data_configs.json'

SEED_NUM = 42
tf.random.set_seed(SEED_NUM)

index_inputs = np.load(open(DATA_IN_PATH + TRAIN_INPUTS, 'rb'))
index_outputs = np.load(open(DATA_IN_PATH + TRAIN_OUTPUTS, 'rb'))
index_targets = np.load(open(DATA_IN_PATH + TRAIN_TARGETS, 'rb'))
prepro_config = json.load(open(DATA_IN_PATH + DATA_CONFIG, 'r'))

MODEL_NAME = 'transformer_kor'
BATCH_SIZE = 24
MAX_SEQUENCE = 25
EPOCH = 30
# UNITS = 1024
# EMBEDDING_DIM = 256
VALIDATION_SPLIT = 0.1

word2idx = prepro_config['word2idx']
idx2word = prepro_config['idx2word']

sos_idx = prepro_config['sos_symbol']
eos_idx = prepro_config['eos_symbol']
vocab_size = prepro_config['vocab_size']

kargs = {'model_name': MODEL_NAME,
         'num_layer': 2,
         'd_model': 512,  # 512 로 시작해서 2048, 그리고 다시 512
         'num_heads': 8,  # 8개로 나눔
         'dff': 2048,
         'input_vocab_size': vocab_size,
         'output_vocab_size': vocab_size,
         # 처음에 input encoding 할때 positional encoding(위치정보) (sin,cos func) max_sequence 만큼 값이 있어야 한다.
         'maximum_position_encoding': MAX_SEQUENCE,
         'end_token_idx': word2idx[eos_idx],
         'rate': 0.1
         }

# 문장의 길이가 MAXLEN 이 안되는 경우가 있다. 나머지 빈공간은 셀프 어텐션을 하면 안되니까 0으로 마스킹 처리한다.


def create_padding_mask(seq):  # 입력값은 [batchsize, seq_len] 차원을 늘려줘야한다. 5행
    # 0은 padding equal 함수를 쓰면 0 이면 True 가 return
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 입력의 차원을 늘려준다, (batchsize,1,1,seqlen)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    # 문장 25개? (25,25)

    # tf.linalg.band_part? ->>
    mask = 1-tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(input, target):
    enc_padding_mask = create_padding_mask(input)
    dec_padding_mask = create_padding_mask(input)  # 멀티헤드 어텐션에서 사용할것임.

    # target (sample, seq-len) 이기때문에 seq_len 을 가져오는것임
    # 과거의 단어로 현제의 단어를 예측.
    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[1])
    dec_target_padding_mask = create_padding_mask(target)

    # 2개를 합친다 maximum [배열a],[배열b] 두 배열의 각각 위치에서 높은값을 가져온다?
    # >> 나중에 combined_mask 차원 디버깅 꼭 해볼것 >> 아마 비교해서 마스크 처리 되어있는 부분은 제외하고 가져오는것으로 예상됨
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# Positional Encoding
# 2가지 구현이 있음. 논문에서는 그냥 수식만 주고 Positional Encoding 을 한다 고만 함
# 수식 캡쳐해서 임베딩할것
def get_angles(pos, i, d_model):
    #angle_rate = 1/np.power(10000,(2*i//2) /np.float(d_model))
    angle_rate = 1/np.power(10000, (2*i//2)/np.float(d_model))
    # print(angle_rate)
    return pos * angle_rate


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # print(angle_rads)
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    # print(angle_rads)

    # ... 은 뭐지 ?
    # 예를들어 A = (50,) 이건 1차원인데 [A,:] 이렇게하면 2차원으로 바뀜 근데 3차원이면 이게 안됨..
    # 우리같은 경우 (50,512) 이므로 이걸 가져간 상태에서 차원을 하나 추가하라는 이야기임
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    # ** 두개면 Dic unpacking
    def __init__(self, **kargs):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = kargs['num_heads']
        self.d_model = kargs['d_model']

        # 0인지 검사한다.
        # 기존에 있는 단어벡터 행렬에서 8개로 잘 나눠야 512가 되는데 나중에 나온 결과를 head 수 만큼 더해줘야 한다
        # 그래서 검사한다. 나누어 떨어지는지 ? 검사
        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(kargs['d_model'])
        self.wk = tf.keras.layers.Dense(kargs['d_model'])
        self.wv = tf.keras.layers.Dense(kargs['d_model'])

        self.dense = tf.keras.layers.Dense(kargs['d_model'])

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))

        # perm 차원을 바꿔준다 [30,1,8,64] >> [30,8,1,64] 변경해준다
        # 이유는 곱해줘야 하기 때문이라고 한다.
        # perm 번호순으로 자리를 바꿔준다고 생각하면됨
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]  # 맨 앞이 batch_size

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wq(k)  # (batch_size, seq_len, d_model)
        v = self.wq(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weight = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weight


enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(
    index_inputs, index_outputs)

#index_inputs, enc_padding_mask
d_model = kargs['d_model']
embedding = tf.keras.layers.Embedding(kargs['input_vocab_size'], d_model)
pos_encoding = positional_encoding(kargs['maximum_position_encoding'], d_model)

seq_len = tf.shape(index_inputs)[1]

x = embedding(index_inputs)
x * tf.math.sqrt(tf.cast(d_model, tf.float32))
x += pos_encoding[:, : seq_len, :]

mha = MultiHeadAttention(**kargs)
# x 하나로 qkv 를 만들어줌
attn_output, _ = mha(x, x, x, enc_padding_mask)
