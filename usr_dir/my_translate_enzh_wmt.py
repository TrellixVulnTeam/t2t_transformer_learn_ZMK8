from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry
 
import tensorflow as tf
 
FLAGS = tf.flags.FLAGS
 
#End-of-sentence marker
EOS = text_encoder.EOS_ID
 
_STAT_MT_URL = "http://data.statmt.org/wmt18/translation-task/"
_ZHEN_TRAIN_DATASETS = [
    [
    _STAT_MT_URL + "train_data.tgz",
    ("20180624.good_zh.sub", "20180624.good_en.sub")
    ],
    ]
# _ZHEN_TRAIN_DATASETS = [
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180624.good_zh.sub", "20180624.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180625.good_zh.sub", "20180625.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180626.good_zh.sub", "20180626.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180627.good_zh.sub", "20180627.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180628.good_zh.sub", "20180628.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180629.good_zh.sub", "20180629.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180630.good_zh.sub", "20180630.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180703.good_zh.sub", "20180703.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180704.good_zh.sub", "20180704.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180705.good_zh.sub", "20180705.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180706.good_zh.sub", "20180706.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180707.good_zh.sub", "20180707.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180708.good_zh.sub", "20180708.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180709.good_zh.sub", "20180709.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180710.good_zh.sub", "20180710.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180711.good_zh.sub", "20180711.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180712.good_zh.sub", "20180712.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180713.good_zh.sub", "20180713.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180714.good_zh.sub", "20180714.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180715.good_zh.sub", "20180715.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180716.good_zh.sub", "20180716.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180717.good_zh.sub", "20180717.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180718.good_zh.sub", "20180718.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180719.good_zh.sub", "20180719.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180720.good_zh.sub", "20180720.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180721.good_zh.sub", "20180721.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180722.good_zh.sub", "20180722.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180723.good_zh.sub", "20180723.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180724.good_zh.sub", "20180724.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180725.good_zh.sub", "20180725.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180726.good_zh.sub", "20180726.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180727.good_zh.sub", "20180727.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180728.good_zh.sub", "20180728.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180729.good_zh.sub", "20180729.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180730.good_zh.sub", "20180730.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("20180731.good_zh.sub", "20180731.good_en.sub")
#     ],
#     [
#     _STAT_MT_URL + "train_data.tgz",
#     ("former.zh.sub", "former.en.sub")
#     ],
#     ]
 
_ZHEN_TEST_DATASETS = [[
    _STAT_MT_URL + "train_data.tgz",
    ("unk.zh.sub", "unk.en.sub")
    ]]
 
def get_filename(dataset):
    return dataset[0][0].split("/")[-1]
 
@registry.register_problem
class MyTranslateEnzhWmt32k(translate.TranslateProblem):
 
    @property
    def is_generate_per_split(self):
        return False
 
    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN
 
    @property
    def oov_token(self):
        return '<unk>'
 
    @property
    def source_vocab_name(self):
        return "zh"
 
    @property
    def target_vocab_name(self):
        return "en"
 
    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return _ZHEN_TRAIN_DATASETS if train else _ZHEN_TEST_DATASETS
 
    def get_training_dataset(self, tmp_dir):
        full_dataset = _ZHEN_TRAIN_DATASETS
        return full_dataset
 
    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        train_dataset = self.get_training_dataset(tmp_dir)
        datasets = self.source_data_files(dataset_split)
        """
        for item in datasets:
            dummy_file_name = item[0].split('/')[-1]
            create_dummy_tar(tmp_dir, dummy_file_name)
            s_file, t_file = item[1][0], item[1][1]
            if not os.path.exists(os.path.join(tmp_dir, s_file)):
                raise Exception("Be sure file '%s' is exists in tmp dir" % s_file)
            if not os.path.exists(os.path.join(tmp_dir, t_file)):
                raise Exception("Be sure file '%s' is exists in tmp dir" % t_file)
        """
        source_datasets = [[item[0], [item[1][0]]] for item in train_dataset]
        target_datasets = [[item[0], [item[1][0]]] for item in train_dataset]
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_encoder = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov=self.oov_token)
        target_encoder = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov=self.oov_token)
        tag = "train" if train else "dev"
        filename_base = "%s-compiled-%s" % (self.name, tag)
        data_path = translate.compile_data(tmp_dir, datasets, filename_base)
        return text_problems.text2text_generate_encoded(
                text_problems.text2text_txt_iterator(data_path + '.lang1',
                                                    data_path + '.lang2'),
                source_encoder, target_encoder)
 
    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov=self.oov_token)
        target_token = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov=self.oov_token)
        return {
                "inputs": source_token,
                "targets": target_token,
        }
 
 
@registry.register_problem
class TranslateEnzhTestWmt8k(TranslateZhenTestWmt32k):
 
    @property
    def approx_vocab_size(self):
        return 2**13
 
    @property
    def dataset_split(self):
        return [
                {
                    "split": problem.DatasetSplit.TRAIN,
                    "shards": 10,
                    },
                {
                    "split": problem.DatasetSplit.EVAL,
                    "shards": 1,
                    }
                ]
 
    def get_training_dataset(self, tmp_dir):
        return _ZHEN_TRAIN_DATASETS
