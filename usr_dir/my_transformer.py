from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.models import transformer


@registry.register_hparams
def my_transformer_base_single_gpu():
"""HParams for transformer base model for single gpu."""
  hparams = transformer.transformer_base()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 16000
  return hparams
