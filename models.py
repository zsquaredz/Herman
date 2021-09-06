import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules import ConditionalRandomField

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

class Encoder_article_RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        # print(text.shape)
        # print(text_lengths.shape)
        # total_length = text.size(0)  # get the max sequence length
        embedded = self.dropout(self.embedding(text))
        # embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]
        # embedded = embedded.permute(1,0,2)
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)

        encoder_outputs = output.contiguous()

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        # hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # hidden = [batch size, hid dim * num directions]

        cell = torch.tanh(self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))

        return encoder_outputs, hidden, cell

class Encoder_article_RNN_BERT(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers,
                 bidirectional, dropout, pad_idx, bert):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        # self.rnn = nn.DataParallel(self.rnn, device_ids=[0, 1])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bert = bert

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        # total_length = text.size(0)  # get the max sequence length
        # embedded = self.dropout(self.embedding(text))
        # embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]
        # embedded = embedded.permute(1,0,2)
        # pack sequence
        text = text.permute(1,0)
        with torch.no_grad():
            embedded = self.dropout(self.bert(text)[0])
            embedded = embedded.permute(1,0,2)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        # unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        encoder_outputs = output.contiguous()

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        # hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))

        # hidden = [batch size, hid dim * num directions]

        cell = torch.tanh(self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))

        return encoder_outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, attention_dim):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim
        self.attn_in = (encoder_hidden_dim * 2) + decoder_hidden_dim
        self.attn = nn.Linear(self.attn_in, attention_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim=2)))

        # attention = torch.sum(energy, dim=2)
        attention = self.v(energy).squeeze(2)
        # attention = [batch size, sent len]
        return F.softmax(attention, dim=1)

class Decoder_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, output_dim, dropout,
                 attention_article):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_article = attention_article
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(encoder_hidden_dim * 2 + embedding_dim, decoder_hidden_dim)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        # self.out = nn.Linear(self.attention_article.attn_in + embedding_dim, output_dim)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):
        a = self.attention_article(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        # a = [batch size, 1, sent len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs =  [batch size, sent len, src hid dim * num directions]
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        # weighted_encoder_rep = [batch size, 1, src hid dim * num directions]
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self, input, decoder_hidden, decoder_cell, encoder_outputs_article):
        input = input.unsqueeze(0) # [1, batch size,
        embedded = self.dropout((self.embedding(input)))
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs_article)
        # weighted_encoder_rep = [1, batch_size, encoder_hidden_dim* 2]
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, (decoder_hidden, decoder_cell) = self.rnn(rnn_input, (decoder_hidden.unsqueeze(0), decoder_cell.unsqueeze(0)))
        # output = [1, batch size, tgt hidden dim]
        output = output.squeeze(0)
        # output = [batch size, tgt hidden dim]

        # embedded = embedded.squeeze(0)
        # weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        # output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        return output, decoder_hidden.squeeze(0), decoder_cell.squeeze(0)

class Decoder_LSTM_BERT(nn.Module):
    def __init__(self, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, output_dim, dropout,
                 attention_article, bert):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_article = attention_article
        self.bert = bert
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(encoder_hidden_dim * 2 + embedding_dim, decoder_hidden_dim)
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        # self.out = nn.Linear(self.attention_article.attn_in + embedding_dim, output_dim)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):
        a = self.attention_article(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        # a = [batch size, 1, sent len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs =  [batch size, sent len, src hid dim * num directions]
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        # weighted_encoder_rep = [batch size, 1, src hid dim * num directions]
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep

    def forward(self, input, decoder_hidden, decoder_cell, encoder_outputs_article):
        input = input.unsqueeze(0) # [1, batch size]
        # embedded = self.dropout((self.embedding(input)))
        input = input.permute(1,0)
        with torch.no_grad():
            embedded = self.dropout(self.bert(input)[0]) # [batch_size, sequence_length, hidden_size]
            embedded = embedded.permute(1, 0, 2)
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs_article)
        # weighted_encoder_rep = [1, batch_size, encoder_hidden_dim* 2]
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)
        output, (decoder_hidden, decoder_cell) = self.rnn(rnn_input, (decoder_hidden.unsqueeze(0), decoder_cell.unsqueeze(0)))
        # output = [1, batch size, tgt hidden dim]
        output = output.squeeze(0)
        # output = [batch size, tgt hidden dim]

        # embedded = embedded.squeeze(0)
        # weighted_encoder_rep = weighted_encoder_rep.squeeze(0)
        # output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))

        return output, decoder_hidden.squeeze(0), decoder_cell.squeeze(0)

class Verification_RNN(nn.Module):
    def __init__(self, encoder_article, encoder_summary, decoder, device, args):
        super().__init__()
        self.encoder_article = encoder_article
        self.encoder_summary = encoder_summary
        self.decoder = decoder
        self.device = device
        self.fc_hidden = nn.Linear(args.src_hidden_dim * 2, args.src_hidden_dim)
        self.fc_cell = nn.Linear(args.src_hidden_dim * 2, args.src_hidden_dim)
        self.fc_binary = nn.Linear(decoder.output_dim, 1)

    def forward(self, article, article_lengths, summary, summary_lengths, summary_label, teacher_forcing_ratio):
        batch_size = article.shape[1]
        output_len = summary.shape[0]
        output_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(output_len, batch_size, output_vocab_size).to(self.device)
        outputs_binary = torch.zeros(output_len, batch_size, 1).to(self.device)
        encoder_outputs_article, hidden_article, cell_article = self.encoder_article(article, article_lengths)
        encoder_outputs_summary, hidden_summary, cell_summary = self.encoder_summary(summary, summary_lengths)

        hidden = self.fc_hidden(torch.cat((hidden_article, hidden_summary), dim=1))
        cell = self.fc_cell(torch.cat((cell_article, cell_summary), dim=1))

        input_token = summary[0, :] # first input to the decoder is the first token in summary
        # input_label = summary_label[0, :] # plus first token in the summary label

        for t in range(0, output_len):
            # TODO: below may be a problem is hidden from encoder and decoder has different dim, maybe add a linear layer
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs_article, encoder_outputs_summary)
            outputs[t] = output # [batch_size, output_vocab_size]
            outputs_binary[t] = self.fc_binary(output) # [batch_size, 1]
            input_token = summary[t] # token will always be using tokens from summary

            # # decide if we are going to use teacher forcing or not
            # teacher_force = random.random() < teacher_forcing_ratio
            # # get the highest predicted token from our predictions
            # top1 = output.argmax(dim=1)
            #
            # # if teacher forcing, use actual next token as next input
            # # if not, use predicted token
            # input_label = summary_label[t] if teacher_force else top1


        # output_binary = torch.sum(outputs_binary, dim=0) # [seq_len, batch_size, 1] -> [batch_size, 1] # no need Sigmoid b/c using BCEWithLogitsLoss
        output_binary = torch.mean(outputs_binary, dim=0)  # [seq_len, batch_size, 1] -> [batch_size, 1] # no need Sigmoid b/c using BCEWithLogitsLoss

        return outputs, output_binary

class Verification_BiLSTM(nn.Module):
    def __init__(self, encoder_article, decoder_LSTM_f, decoder_LSTM_b, device, PAD_IDX):
        super().__init__()
        self.encoder_article = encoder_article
        self.decoder_f = decoder_LSTM_f # forward LSTM
        self.decoder_b = decoder_LSTM_b # backward LSTM
        self.device = device
        self.target_size = decoder_LSTM_f.output_dim  # output_dim
        # Maps the output of the LSTM into tag space
        self.hidden2tag = nn.Linear(decoder_LSTM_f.decoder_hidden_dim*2, self.target_size)
        # Maps the output of the LSTM into binary label
        self.hidden2binary = nn.Linear(decoder_LSTM_f.decoder_hidden_dim*2, 1)
        # self.tag_to_ix = tag_to_ix
        # self.START_IDX = self.target_size - 2
        # self.STOP_IDX = self.target_size - 1
        self.PAD_IDX = PAD_IDX


        # # Matrix of transition parameters.  Entry i,j is the score of
        # # transitioning *to* i *from* j.
        # self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size), requires_grad=True)
        #
        # # These two statements enforce the constraint that we never transfer
        # # to the start tag and we never transfer from the stop tag
        # self.transitions.data[self.START_IDX, :] = -10000.
        # self.transitions.data[:, self.STOP_IDX] = -10000.
        # # self.transitions.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        # # self.transitions.data[PAD_IDX, :] = -10000 # no transition to PAD except from STOP
        # # self.transitions.data[PAD_IDX, self.STOP_IDX] = 0.
        # # self.transitions.data[PAD_IDX, PAD_IDX] = 0.

    def _get_lstm_features(self, article, article_lengths, summary, summary_lengths, mask):
        batch_size = article.shape[1]
        output_len = summary.shape[0]
        output_vocab_size = self.target_size

        # outputs = torch.zeros(output_len, batch_size, output_vocab_size).to(self.device)
        # outputs_binary = torch.zeros(output_len, batch_size, 1).to(self.device)
        output_forward = torch.zeros(output_len, batch_size, self.decoder_f.decoder_hidden_dim).to(self.device)
        output_backward = torch.zeros(output_len, batch_size, self.decoder_b.decoder_hidden_dim).to(self.device)
        encoder_outputs_article, hidden_article, cell_article = self.encoder_article(article, article_lengths)

        hidden_f, cell_f = hidden_article, cell_article
        hidden_b, cell_b = hidden_article, cell_article
        input_token_f = summary[0, :]  # first input to the forward decoder is the first token in summary
        input_token_b = summary[-1, :]  # first input to the backward decoder is the first token in summary
        # one pass to compute both forward and background
        for index_f in range(0, output_len):
            index_b = output_len - index_f - 1 # this is the index for the backward pass
            output_f, hidden_f, cell_f = self.decoder_f(input_token_f, hidden_f, cell_f, encoder_outputs_article)
            output_b, hidden_b, cell_b = self.decoder_b(input_token_b, hidden_b, cell_b, encoder_outputs_article)
            output_forward[index_f] = output_f  # fill this in 0 -> seq_len (left2right)
            output_backward[index_b] = output_b  # fill this in seq_len -> 0 (right2left)
            input_token_f = summary[index_f]  # token will always be using tokens from summary
            input_token_b = summary[index_b]
        # output combined from both directions
        outputs_combined = torch.cat((output_forward, output_backward), dim=2)  # [seq_len, batch_size, hidden_dim *2]
        # map it to output_dim
        outputs = self.hidden2tag(outputs_combined)  # [seq_len, batch_size, output_dim]
        outputs_binary = self.hidden2binary(outputs_combined)  # [seq_len, batch_size, 1]
        # mask = mask.permute(1, 0) # [seq_len, batch_size]
        # outputs *= mask.unsqueeze(2)
        # [seq_len, batch_size, 1] -> [batch_size, 1] # no need Sigmoid b/c using BCEWithLogitsLoss
        output_binary = torch.mean(outputs_binary, dim=0)

        return outputs, output_binary

    def forward(self, article, article_lengths, summary, summary_lengths):
        # this function return both the loss and predicted tags
        mask = (summary != self.PAD_IDX).float() # value is 0 if PAD, value if 1 for normal tokens
        mask = mask.permute(1, 0)  # [batch_size, seq_len]
        # Get the emission scores from the BiLSTM
        lstm_feats, outputs_binary = self._get_lstm_features(article, article_lengths, summary, summary_lengths, mask)
        return lstm_feats, outputs_binary

class Verification_BiLSTM_CRF_allennlp(nn.Module):
    def __init__(self, encoder_article, decoder_LSTM_f, decoder_LSTM_b, device, PAD_IDX):
        super().__init__()
        self.encoder_article = encoder_article
        self.decoder_f = decoder_LSTM_f # forward LSTM
        self.decoder_b = decoder_LSTM_b # backward LSTM
        self.device = device
        self.target_size = decoder_LSTM_f.output_dim  # output_dim
        # Maps the output of the LSTM into tag space
        self.hidden2tag = nn.Linear(decoder_LSTM_f.decoder_hidden_dim*2+1, self.target_size)
        # binary MLP
        self.fc1 = nn.Linear(decoder_LSTM_f.decoder_hidden_dim*2,decoder_LSTM_f.decoder_hidden_dim*2)
        self.fc2 = nn.Linear(decoder_LSTM_f.decoder_hidden_dim*2,decoder_LSTM_f.decoder_hidden_dim*2)
        # Maps the output of the LSTM into binary label
        self.hidden2binary = nn.Linear(decoder_LSTM_f.decoder_hidden_dim*2, 1)
        # self.hidden2binary = nn.Linear(self.target_size, 1)
        # self.tag_to_ix = tag_to_ix
        # self.START_IDX = self.target_size - 2
        # self.STOP_IDX = self.target_size - 1
        self.PAD_IDX = PAD_IDX


        # # Matrix of transition parameters.  Entry i,j is the score of
        # # transitioning *to* i *from* j.
        # self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size), requires_grad=True)
        #
        # # These two statements enforce the constraint that we never transfer
        # # to the start tag and we never transfer from the stop tag
        # self.transitions.data[self.START_IDX, :] = -10000.
        # self.transitions.data[:, self.STOP_IDX] = -10000.
        # # self.transitions.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        # # self.transitions.data[PAD_IDX, :] = -10000 # no transition to PAD except from STOP
        # # self.transitions.data[PAD_IDX, self.STOP_IDX] = 0.
        # # self.transitions.data[PAD_IDX, PAD_IDX] = 0.

        # List[Tuple[int, int]]: An optional list of allowed transitions (from_tag_id, to_tag_id)
        # {'<pad>': 0, 'O': 1, 'BV': 2, 'BU': 3, 'IV': 4, 'IU': 5, 'START': 6, 'END': 7}
        constraints = [(0,0),
                       (7,0),
                       (6,1),(6,2),(6,3),
                       (1,1),(1,2),(1,3),(1,7),
                       (2,1),(2,2),(2,3),(2,4),(2,7),
                       (3,1),(3,2),(3,3),(3,5),(3,7),
                       (4,1),(4,2),(4,3),(4,4),(4,7),
                       (5,1),(5,2),(5,3),(5,5),(5,7)]
        self.crf = ConditionalRandomField(self.target_size, constraints=constraints)

    def _get_lstm_features(self, article, article_lengths, summary, summary_lengths, mask, quantity_label):
        batch_size = article.shape[1]
        output_len = summary.shape[0]
        # output_vocab_size = self.target_size

        # outputs = torch.zeros(output_len, batch_size, output_vocab_size).to(self.device)
        # outputs_binary = torch.zeros(output_len, batch_size, 1).to(self.device)
        output_forward = torch.zeros(output_len, batch_size, self.decoder_f.decoder_hidden_dim).to(self.device)
        output_backward = torch.zeros(output_len, batch_size, self.decoder_b.decoder_hidden_dim).to(self.device)
        encoder_outputs_article, hidden_article, cell_article = self.encoder_article(article, article_lengths)

        hidden_f, cell_f = hidden_article, cell_article
        hidden_b, cell_b = hidden_article, cell_article
        input_token_f = summary[0, :]  # first input to the forward decoder is the first token in summary
        input_token_b = summary[-1, :]  # first input to the backward decoder is the first token in summary
        # one pass to compute both forward and background
        for index_f in range(0, output_len):
            index_b = output_len - index_f - 1 # this is the index for the backward pass
            output_f, hidden_f, cell_f = self.decoder_f(input_token_f, hidden_f, cell_f, encoder_outputs_article)
            output_b, hidden_b, cell_b = self.decoder_b(input_token_b, hidden_b, cell_b, encoder_outputs_article)
            output_forward[index_f] = output_f  # fill this in 0 -> seq_len (left2right)
            output_backward[index_b] = output_b  # fill this in seq_len -> 0 (right2left)
            input_token_f = summary[index_f]  # token will always be using tokens from summary
            input_token_b = summary[index_b]
        # output combined from both directions
        outputs_combined = torch.cat((output_forward, output_backward), dim=2)  # [seq_len, batch_size, hidden_dim *2]
        # add quantity label information
        quantity_label = quantity_label.unsqueeze(2)
        # [seq_len, batch_size, 1]
        outputs_combined1 = torch.cat((outputs_combined, quantity_label), dim=2)
        # map it to output_dim
        outputs = self.hidden2tag(outputs_combined1)  # [seq_len, batch_size, output_dim]
        # outputs_combined = F.relu(self.fc1(outputs_combined))
        outputs_combined = F.relu(self.fc2(outputs_combined))
        outputs_binary = self.hidden2binary(outputs_combined)  # [seq_len, batch_size, 1]
        # outputs_binary = self.hidden2binary(outputs)  # [seq_len, batch_size, 1] instead of using raw features, use directly outputs
        # mask = mask.permute(1, 0) # [seq_len, batch_size]
        # outputs *= mask.unsqueeze(2)
        # [seq_len, batch_size, 1] -> [batch_size, 1] # no need Sigmoid b/c using BCEWithLogitsLoss
        output_binary = torch.mean(outputs_binary, dim=0)

        return outputs, output_binary

    def forward(self, article, article_lengths, summary, summary_lengths, quantity_label):
        # this function return both the loss and predicted tags
        mask = (summary != self.PAD_IDX) # value is 0 if PAD, value if 1 for normal tokens
        mask = mask.permute(1, 0)  # [batch_size, seq_len]
        # Get the emission scores from the BiLSTM
        lstm_feats, outputs_binary = self._get_lstm_features(article, article_lengths, summary, summary_lengths, mask, quantity_label)
        lstm_feats = lstm_feats.permute(1,0,2) # [batch_size, seq_len, output_dim]
        tag_seq = self.crf.viterbi_tags(logits=lstm_feats, mask=mask)

        return tag_seq, outputs_binary, lstm_feats

    def neg_log_likelihood(self, article, article_lengths, summary, summary_lengths, tags, quantity_label):
        mask = (summary != self.PAD_IDX)  # value is 0 if PAD, value if 1 for normal tokens
        mask = mask.permute(1, 0)  # [batch_size, seq_len]
        lstm_feats, outputs_binary = self._get_lstm_features(article, article_lengths, summary, summary_lengths, mask, quantity_label)
        lstm_feats = lstm_feats.permute(1, 0, 2)  # [batch_size, seq_len, hidden_dim]
        tags = tags.permute(1, 0)  # [batch_size, seq_len]
        loss = self.crf(inputs=lstm_feats, tags=tags, mask=mask) # note this loss is (joint - input) thus need to *-1
        return -1*loss, outputs_binary

class Verification_BiLSTM_CRF(nn.Module):
    def __init__(self, encoder_article, decoder_LSTM_f, decoder_LSTM_b, device, PAD_IDX):
        super().__init__()
        self.encoder_article = encoder_article
        self.decoder_f = decoder_LSTM_f # forward LSTM
        self.decoder_b = decoder_LSTM_b # backward LSTM
        self.device = device
        self.target_size = decoder_LSTM_f.output_dim + 2  # output_dim + <START> <STOP>
        # Maps the output of the LSTM into tag space
        self.hidden2tag = nn.Linear(decoder_LSTM_f.decoder_hidden_dim*2+1, self.target_size)
        # Maps the output of the LSTM into binary label
        self.hidden2binary = nn.Linear(decoder_LSTM_f.decoder_hidden_dim*2+1, 1)
        # self.tag_to_ix = tag_to_ix
        self.START_IDX = self.target_size - 2
        self.STOP_IDX = self.target_size - 1
        self.PAD_IDX = PAD_IDX


        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size), requires_grad=True)
        self.transitions = nn.Parameter(torch.Tensor(self.target_size, self.target_size), requires_grad=True)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        # self.transitions.data[self.START_IDX, :] = -10000.
        # self.transitions.data[:, self.STOP_IDX] = -10000.
        # self.transitions.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        # self.transitions.data[PAD_IDX, :] = -10000 # no transition to PAD except from STOP
        # self.transitions.data[PAD_IDX, self.STOP_IDX] = 0.
        # self.transitions.data[PAD_IDX, PAD_IDX] = 0.

        # transition mask Tij -> (j -> i) (to_tag_id, from_tag_id) [PAD O BV BU IV IU START END]
        constraints = [[1,0,0,0,0,0,0,1],
                       [0,1,1,1,1,1,1,0],
                       [0,1,1,1,1,1,1,0],
                       [0,1,1,1,1,1,1,0],
                       [0,0,1,0,1,0,0,0],
                       [0,0,0,1,0,1,0,0],
                       [0,0,0,0,0,0,0,0],
                       [0,1,1,1,1,1,0,0]]
        # constraints
        constraint_mask = torch.Tensor(constraints)
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.transitions)

    def _get_lstm_features(self, article, article_lengths, summary, summary_lengths, mask, quantity_label):
        batch_size = article.shape[1]
        output_len = summary.shape[0]
        output_vocab_size = self.target_size

        # outputs = torch.zeros(output_len, batch_size, output_vocab_size).to(self.device)
        output_forward = torch.zeros(output_len, batch_size, self.decoder_f.decoder_hidden_dim).to(self.device)
        output_backward = torch.zeros(output_len, batch_size, self.decoder_b.decoder_hidden_dim).to(self.device)
        # outputs_binary = torch.zeros(output_len, batch_size, 1).to(self.device)
        encoder_outputs_article, hidden_article, cell_article = self.encoder_article(article, article_lengths)

        hidden_f, cell_f = hidden_article, cell_article
        hidden_b, cell_b = hidden_article, cell_article
        input_token_f = summary[0, :]  # first input to the forward decoder is the first token in summary
        input_token_b = summary[-1, :]  # first input to the backward decoder is the first token in summary
        # one pass to compute both forward and background
        for index_f in range(0, output_len):
            index_b = output_len - index_f - 1 # this is the index for the backward pass
            output_f, hidden_f, cell_f = self.decoder_f(input_token_f, hidden_f, cell_f, encoder_outputs_article)
            output_b, hidden_b, cell_b = self.decoder_b(input_token_b, hidden_b, cell_b, encoder_outputs_article)

            output_forward[index_f] = output_f # fill this in 0 -> seq_len (left2right)
            output_backward[index_b] = output_b # fill this in seq_len -> 0 (right2left)
            # output = torch.cat((output_f, output_b), dim=1) # [batch_size, decoder_hidden_dim * 2]
            # output combined from both directions
            # outputs[index_f] = self.hidden2tag(output)  # [batch_size, output_dim]
            # outputs_binary[index_f] = self.hidden2binary(output)  # [batch_size, 1]

            input_token_f = summary[index_f]  # token will always be using tokens from summary
            input_token_b = summary[index_b]
        # output combined from both directions
        outputs_combined = torch.cat((output_forward, output_backward), dim=2) # [seq_len, batch_size, hidden_dim *2]
        # add quantity label information
        quantity_label = quantity_label.unsqueeze(2)  # [seq_len, batch_size, 1]
        outputs_combined = torch.cat((outputs_combined, quantity_label), dim=2)
        # map it to output_dim
        outputs = self.hidden2tag(outputs_combined) # [seq_len, batch_size, output_dim]
        outputs_binary = self.hidden2binary(outputs_combined) # [seq_len, batch_size, 1]


        # mask = mask.permute(1, 0) # [seq_len, batch_size]
        # outputs *= mask.unsqueeze(2)
        # [seq_len, batch_size, 1] -> [batch_size, 1] # no need Sigmoid b/c using BCEWithLogitsLoss
        output_binary = torch.mean(outputs_binary, dim=0)

        return outputs, output_binary

    def _forward_alg(self, feats, mask):
        feats = feats.permute(1,0,2) # [batch_size, seq_len, output_dim]
        batch_size = feats.size(0)
        # Do the forward algorithm to compute the partition function
        scores = torch.full((batch_size, self.target_size), -10000., device=self.device)
        # START_TAG has all of the score.
        scores[:, self.START_IDX] = 0.
        trans = self.transitions.unsqueeze(0) # [1, output_dim, output_dim] output_dim aka target_size
        # Wrap in a variable so that we will get automatic backprop
        # scores = scores.to(self.device)
        # Iterate through the sentence
        for t in range(feats.size(1)):
            mask_t = mask[:,t].unsqueeze(1)
            emit_t = feats[:,t].unsqueeze(2) # [batch_size, output_dim, 1]
            score_t = scores.unsqueeze(1) + emit_t + trans # [batch_size, output_dim, output_dim]
            score_t = log_sum_exp(score_t) # [batch_size, output_dim]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.STOP_IDX])
        return scores

    def _score_sentence(self, feats, tags, mask):
        feats = feats.permute(1, 0, 2)  # [batch_size, seq_len, output_dim]
        tags = tags.permute(1, 0) # [batch_size, seq_len]
        batch_size = feats.size(0)
        # Gives the score of a provided tag sequence

        emit_scores = feats.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)
        # transition score
        start_tag = torch.full((batch_size, 1), self.START_IDX, dtype=torch.long).to(self.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [batch_size, seq_len+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=mask.sum(1).long().unsqueeze(1)).squeeze(1)  # [batch_size]
        last_score = self.transitions[self.STOP_IDX, last_tag]

        score = ((trans_scores + emit_scores) * mask).sum(1) + last_score
        return score

    def _viterbi_decode(self, feats, mask):
        features = feats.permute(1, 0, 2)  # [batch_size, seq_len, output_dim]
        # batch_size = feats.size(0)

        B, L, C = features.shape # [batch_size, seq_len, output_dim]

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), -10000., device=features.device)  # [batch_size, output_dim]
        max_score[:, self.START_IDX] = 0.

        # Apply transition constraints
        constrained_transitions = self.transitions * self._constraint_mask - 1000000. * (1 - self._constraint_mask)
        # print('---')
        # print(self.transitions)
        # print(constrained_transitions)

        for t in range(L):
            mask_t = mask[:, t].unsqueeze(1)  # [batch_size, 1]
            emit_score_t = features[:, t]  # [batch_size, output_dim]

            # [batch_size, 1, output_dim] + [output_dim, output_dim]
            # acc_score_t = max_score.unsqueeze(1) + self.transitions  # [batch_size, output_dim, output_dim]
            acc_score_t = max_score.unsqueeze(1) + constrained_transitions  # [batch_size, output_dim, output_dim]
            # print(acc_score_t)
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        # max_score += self.transitions[self.STOP_IDX]
        max_score += constrained_transitions[self.STOP_IDX]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(mask[b, :].sum().item())
            best_path = [best_tag_b] + [bps_t[best_tag_b] for bps_t in reversed(bps[b, :seq_len])]
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])
        return best_score, best_paths

    def neg_log_likelihood(self, article, article_lengths, summary, summary_lengths, tags, quantity_label):
        mask = (summary != self.PAD_IDX).float()  # value is 0 if PAD, value if 1 for normal tokens
        mask = mask.permute(1, 0) # [batch_size, seq_len]
        feats, outputs_binary = self._get_lstm_features(article, article_lengths, summary, summary_lengths, mask, quantity_label)
        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)
        return (forward_score - gold_score).mean(), outputs_binary

    def decode(self, article, article_lengths, summary, summary_lengths, quantity_label):  # dont confuse this with _forward_alg above.
        # this function only do the viterbi and output tags, no loss will be calc
        mask = (summary != self.PAD_IDX).float() # value is 0 if PAD, value if 1 for normal tokens
        mask = mask.permute(1, 0)  # [batch_size, seq_len]
        # Get the emission scores from the BiLSTM
        lstm_feats, outputs_binary = self._get_lstm_features(article, article_lengths, summary, summary_lengths, mask, quantity_label)

        # Find the best path, given the features.
        _, tag_seq = self._viterbi_decode(lstm_feats, mask)
        return tag_seq, outputs_binary

    def forward(self, article, article_lengths, summary, summary_lengths, tags, quantity_label):  # dont confuse this with _forward_alg above.
        # this function return both the loss and predicted tags
        mask = (summary != self.PAD_IDX).float() # value is 0 if PAD, value if 1 for normal tokens
        mask = mask.permute(1, 0)  # [batch_size, seq_len]
        # Get the emission scores from the BiLSTM
        lstm_feats, outputs_binary = self._get_lstm_features(article, article_lengths, summary, summary_lengths, mask, quantity_label)
        forward_score = self._forward_alg(lstm_feats, mask)
        gold_score = self._score_sentence(lstm_feats, tags, mask)
        # Find the best path, given the features.
        _, tag_seq = self._viterbi_decode(lstm_feats, mask)
        return (forward_score - gold_score).mean(), tag_seq, outputs_binary
