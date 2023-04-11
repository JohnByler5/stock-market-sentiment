import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import LSTM

import flair.embeddings
import flair.nn
from flair.data import DT, DT2, Dictionary, Sentence, Span, Token
from flair.embeddings import Embeddings

log = logging.getLogger("flair")


class SimpleSequenceTagger(flair.nn.DefaultClassifier[Sentence, Span]):
    def __init__(
        self,
        embeddings: flair.embeddings.TokenEmbeddings,
        tag_dictionary: Dictionary,
        tag_type: str,
        tag_format: str = "BIOES",
        use_rnn: bool = True,
        **classifierargs,
    ):
        # span-labels need special encoding (BIO or BIOES)
        if tag_dictionary.span_labels:
            # the big question is whether the label dictionary should contain an UNK or not
            # without UNK, we cannot evaluate on data that contains labels not seen in test
            # with UNK, the model learns less well if there are no UNK examples
            label_dictionary = Dictionary(add_unk=False)
            assert tag_format in ["BIOES", "BIO"]
            for label in tag_dictionary.get_items():
                if label == "<unk>":
                    continue
                label_dictionary.add_item("O")
                if tag_format == "BIOES":
                    label_dictionary.add_item("S-" + label)
                    label_dictionary.add_item("B-" + label)
                    label_dictionary.add_item("E-" + label)
                    label_dictionary.add_item("I-" + label)
                if tag_format == "BIO":
                    label_dictionary.add_item("B-" + label)
                    label_dictionary.add_item("I-" + label)

            self.predict_spans = True
        else:
            label_dictionary = tag_dictionary

        super().__init__(
            label_dictionary=label_dictionary,
            final_embedding_size=256 * 2 if use_rnn else embeddings.embedding_length,
            **classifierargs,
        )

        self._label_type = tag_type
        self.embeddings = embeddings
        self.tag_format = tag_format

        if use_rnn:
            self.rnn = LSTM(hidden_size=256, input_size=embeddings.embedding_length, bidirectional=True)
        else:
            self.rnn = None

        # all parameters will be pushed internally to the specified device
        self.to(flair.device)

    def _get_embedding_for_data_point(self, prediction_data_point: Token) -> torch.Tensor:
        names = self.embeddings.get_names()
        return prediction_data_point.get_embedding(names)

    def _get_prediction_data_points(self, sentences: List[Sentence]) -> List[Token]:
        tokens: List[Token] = []
        for sentence in sentences:
            tokens.extend(sentence.tokens)
        return tokens

    def _transform_embeddings(
        self,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        """This method does applies a transformation through the model given a list of tensors as input.
        Returns the embeddings that will ran trough a linear decoder layer to calculate logits.

        If it is not overwritten, it will act as identity function for of the first tensor.
        """
        if not self.rnn:
            return args[0]
        # print(args[0].size())
        packed = torch.nn.utils.rnn.pack_sequence(args[0].split(args[-1].tolist()), enforce_sorted=False)
        rnn_output, hidden = self.rnn(packed)
        rnn_output = torch.cat(torch.nn.utils.rnn.unpack_sequence(rnn_output))
        return rnn_output

    @property
    def label_type(self):
        return self._label_type

    @property
    def _inner_embeddings(self) -> Embeddings[Token]:
        return self.embeddings
