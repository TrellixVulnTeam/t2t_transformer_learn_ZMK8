# -*- coding: utf-8 -*-
# @Time    : 18-10-1 下午3:13
# @Author  : lixiaolong2
# @File    : my_translate_ende_bpe32k.py
"""自定义数据处理学习"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

_ENDE_TRAIN_DATASETS = [
    [
        "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz",  # pylint: disable=line-too-long
        ("training-parallel-nc-v13/news-commentary-v13.de-en.en",
         "training-parallel-nc-v13/news-commentary-v13.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.de-en.en", "commoncrawl.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.de-en.en", "training/europarl-v7.de-en.de")
    ],
]
_ENDE_TEST_DATASETS = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.de")
    ],
]

def get_enzh_bpe_dataset(directory, filename):
    train_path = os.path.join(directory, filename)
    if not (tf.gfile.Exists(train_path + ".en") and
            tf.gfile.Exists(train_path + ".zh")):
        raise Exception("there should be some training/dev data in the tmp dir.")

    return train_path

@registry.register_problem
class MyTranslateEndeBpe32K(translate.TranslateProblem):
    @property
    def approx_vocab_size(self):
        return 2*15 # 32k

    @property
    def source_vocab_name(self):
        return "vocab.bpe.en.%d" %self.approx_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.bpe.de.%d" %self.approx_vocab_size

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        """Instance of token generator for the WMT en->zh task, training set."""
        train = dataset_split == problem.DatasetSplit.TRAIN
        dataset_path = (ENZH_BPE_DATASETS["TRAIN"] if train else ENZH_BPE_DATASETS["DEV"])
        train_path = get_enzh_bpe_dataset(tmp_dir, dataset_path)

        # Vocab
        src_token_path = (os.path.join(data_dir, self.source_vocab_name), self.source_vocab_name)
        tar_token_path = (os.path.join(data_dir, self.target_vocab_name), self.target_vocab_name)
        for token_path, vocab_name in [src_token_path, tar_token_path]:
            if not tf.gfile.Exists(token_path):
                token_tmp_path = os.path.join(tmp_dir, vocab_name)
                tf.gfile.Copy(token_tmp_path, token_path)
                with tf.gfile.GFile(token_path, mode="r") as f:
                    vocab_data = "<pad>\n<EOS>\n" + f.read() + "UNK\n"
                with tf.gfile.GFile(token_path, mode="w") as f:
                    f.write(vocab_data)

        return text_problems.text2text_txt_iterator(train_path + ".en",
                                                    train_path + ".zh")

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        """在生成数据的时候，主要是通过这个方法获取已编码样本的"""
        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_vocab(data_dir)
        target_encoder = self.get_vocab(data_dir, is_target=True)
        return text_problems.text2text_generate_encoded(generator, encoder, target_encoder,
                                                        has_inputs=self.has_inputs)

    def feature_encoders(self, data_dir):
        source_token = self.get_vocab(data_dir)
        target_token = self.get_vocab(data_dir, is_target=True)
        return {
            "inputs": source_token,
            "targets": target_token,
        }

