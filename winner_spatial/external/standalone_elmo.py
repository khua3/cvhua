
import json

import h5py
import numpy as np
import torch
import torch.nn as nn


def _make_bos_eos(
    character: int,
    padding_character: int,
    beginning_of_word_character: int,
    end_of_word_character: int,
    max_word_length: int,
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoCharacterMapper:

    max_word_length = 50
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260  # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )

    bos_token = "<S>"
    eos_token = "</S>"

    def __init__(self, tokens_to_add = None) -> None:
        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str):
        if word in self.tokens_to_add:
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = ELMoCharacterMapper.end_of_word_character
        elif word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded = word.encode("utf-8", "ignore")[
                : (ELMoCharacterMapper.max_word_length - 2)
            ]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


class Highway(nn.Module):

    def __init__(
        self,
        input_dim,
        num_layers = 1,
        activation = torch.nn.functional.relu,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class ElmoCharacterEncoder(nn.Module):

    def __init__(self, options_file: str, weight_file: str, requires_grad: bool = False) -> None:
        super().__init__()

        with open(options_file, "r") as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file

        self.output_dim = self._options["lstm"]["projection_dim"]
        self.requires_grad = requires_grad

        self._load_weights()

        self._beginning_of_sentence_characters = torch.from_numpy(
            np.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = torch.from_numpy(
            np.array(ELMoCharacterMapper.end_of_sentence_characters) + 1
        )

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):
        mask = (inputs > 0).sum(dim=-1) > 0
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids( inputs, mask, self._beginning_of_sentence_characters, self._end_of_sentence_characters)
        # the character id embedding
        max_chars_per_token = self._options["char_cnn"]["max_characters_per_token"]
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
            character_ids_with_bos_eos.view(-1, max_chars_per_token), self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options["char_cnn"]
        if cnn_options["activation"] == "tanh":
            activation = torch.tanh
        elif cnn_options["activation"] == "relu":
            activation = torch.nn.functional.relu
        else:
            raise Exception("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, "char_conv_{}".format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return {
            "mask": mask_with_bos_eos,
            "token_embedding": token_embedding.view(batch_size, sequence_length, -1),
        }

    def _load_weights(self):
        self._load_char_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_char_embedding(self):
        with h5py.File(self._weight_file, "r") as fin:
            char_embed_weights = fin["char_embed"][...]

        weights = np.zeros((char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]), dtype="float32")
        weights[1:, :] = char_embed_weights

        self._char_embedding_weights = torch.nn.Parameter(torch.FloatTensor(weights), requires_grad=self.requires_grad )

    def _load_cnn_weights(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        char_embed_dim = cnn_options["embedding"]["dim"]

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim, out_channels=num, kernel_size=width, bias=True
            )
            # load the weights
            with h5py.File(self._weight_file, "r") as fin:
                weight = fin["CNN"]["W_cnn_{}".format(i)][...]
                bias = fin["CNN"]["b_cnn_{}".format(i)][...]

            w_reshaped = np.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError("Invalid weight file")
            conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
            conv.bias.data.copy_(torch.FloatTensor(bias))

            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad

            convolutions.append(conv)
            self.add_module("char_conv_{}".format(i), conv)

        self._convolutions = convolutions

    def _load_highway(self):

        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options["n_highway"]

        # create the layers, and load the weights
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multiplication with concatenation of
            # transform and carry weights.
            with h5py.File(self._weight_file, "r") as fin:
                # The weights are transposed due to multiplication order assumptions in tf
                # vs cliora (tf.matmul(X, W) vs cliora.matmul(W, X))
                w_transform = np.transpose(fin["CNN_high_{}".format(k)]["W_transform"][...])
                # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                w_carry = -1.0 * np.transpose(fin["CNN_high_{}".format(k)]["W_carry"][...])
                weight = np.concatenate([w_transform, w_carry], axis=0)
                self._highways._layers[k].weight.data.copy_(torch.FloatTensor(weight))
                self._highways._layers[k].weight.requires_grad = self.requires_grad

                b_transform = fin["CNN_high_{}".format(k)]["b_transform"][...]
                b_carry = -1.0 * fin["CNN_high_{}".format(k)]["b_carry"][...]
                bias = np.concatenate([b_transform, b_carry], axis=0)
                self._highways._layers[k].bias.data.copy_(torch.FloatTensor(bias))
                self._highways._layers[k].bias.requires_grad = self.requires_grad

    def _load_projection(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)

        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        with h5py.File(self._weight_file, "r") as fin:
            weight = fin["CNN_proj"]["W_proj"][...]
            bias = fin["CNN_proj"]["b_proj"][...]
            self._projection.weight.data.copy_(torch.FloatTensor(np.transpose(weight)))
            self._projection.bias.data.copy_(torch.FloatTensor(bias))

            self._projection.weight.requires_grad = self.requires_grad
            self._projection.bias.requires_grad = self.requires_grad


class SimpleTokenIndexer:
    def __init__(self, tokens_to_add=None, token_min_padding_length=0):
        self._mapper = ELMoCharacterMapper(tokens_to_add)
        self._token_min_padding_length = token_min_padding_length

    def get_padding_lengths(self, indexed_tokens):
        """
        This method returns a padding dictionary for the given `indexed_tokens` specifying all
        lengths that need padding.  If all you have is a list of single ID tokens, this is just the
        length of the list, and that's what the default implementation will give you.  If you have
        something more complicated, like a list of character ids for token, you'll need to override
        this.
        """
        padding_lengths = []
        for token_list in indexed_tokens:
            padding_lengths.append(max(len(token_list), self._token_min_padding_length))
        return padding_lengths

    def as_padded_tensor(self, tokens, padding_lengths):
        def padding_token():
            return [0] * ELMoCharacterMapper.max_word_length

        tokens_tensor = torch.LongTensor(
            pad_sequence_to_length(
                tokens, padding_lengths, default_value=padding_token
            )
        )
        return tokens_tensor


def pad_sequence_to_length(
    sequence,
    desired_length: int,
    default_value = lambda: 0,
    padding_on_right: bool = True,
):
    sequence = list(sequence)
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    pad_length = desired_length - len(padded_sequence)
    # This just creates the default value once, so if it's a list, and if it gets mutated
    # later, it could cause subtle bugs. But the risk there is low, and this is much faster.
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence


def remove_sentence_boundaries(tensor, mask):
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] - 2
    tensor_without_boundary_tokens = tensor.new_zeros(*new_shape)
    new_mask = tensor.new_zeros((new_shape[0], new_shape[1]), dtype=torch.bool)
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            tensor_without_boundary_tokens[i, : (j - 2), :] = tensor[i, 1 : (j - 1), :]
            new_mask[i, : (j - 2)] = True

    return tensor_without_boundary_tokens, new_mask


def batch_to_ids(batch):
    tokens = []
    indexer = SimpleTokenIndexer()

    for sentence in batch:
        char_tokens_list = [indexer._mapper.convert_word_to_char_ids(word) for word in sentence]
        tokens.append(char_tokens_list)

    padding_lengths = indexer.get_padding_lengths(tokens)
    desired_length = max(padding_lengths)
    tokens_tensor = torch.stack([indexer.as_padded_tensor(x, desired_length) for x in tokens])

    return tokens_tensor


def add_sentence_boundary_token_ids(tensor, mask, sentence_begin_token, sentence_end_token):
    sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
    tensor_shape = list(tensor.data.shape)
    new_shape = list(tensor_shape)
    new_shape[1] = tensor_shape[1] + 2
    tensor_with_boundary_tokens = tensor.new_zeros(*new_shape, device=tensor.device)
    if len(tensor_shape) == 2:
        tensor_with_boundary_tokens[:, 1:-1] = tensor
        tensor_with_boundary_tokens[:, 0] = sentence_begin_token
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, j + 1] = sentence_end_token
        new_mask = tensor_with_boundary_tokens != 0
    elif len(tensor_shape) == 3:
        tensor_with_boundary_tokens[:, 1:-1, :] = tensor
        sentence_begin_token = sentence_begin_token.detach().to(tensor.device)
        sentence_end_token = sentence_end_token.detach().to(tensor.device)
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_tokens[i, 0, :] = sentence_begin_token
            tensor_with_boundary_tokens[i, j + 1, :] = sentence_end_token
        new_mask = (tensor_with_boundary_tokens > 0).sum(dim=-1) > 0
    else:
        raise ValueError("add_sentence_boundary_token_ids only accepts 2D and 3D input")

    return tensor_with_boundary_tokens, new_mask