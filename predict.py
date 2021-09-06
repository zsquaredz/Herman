
from torchtext import data
import torch.optim as optim
import logging
from tqdm import tqdm
import argparse
from models import *
import math

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO, datefmt='%b-%d-%Y %H:%M:%S')
parser = argparse.ArgumentParser()
parser.add_argument('--task', default='test', help='test')
parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs.')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--checkpoint', default='checkpoint', help='name of saved checkpoint')
parser.add_argument('--input_dim', type=int, default=500002, help='dimension of input')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimension')
parser.add_argument('--src_hidden_dim', type=int, default=256, help='encoder hidden dimension')
parser.add_argument('--tgt_hidden_dim', type=int, default=256, help='decoder hidden dimension')
parser.add_argument('--output_dim', type=int, default=6, help='output dimension')
parser.add_argument('--attention_dim', type=int, default=256, help='attention dimension')
parser.add_argument('--src_num_layers', type=int, default=2, help='encoder number layers')
parser.add_argument('--src_bidirectional', type=str2bool, default=True, help='encoder bidirectional?')
parser.add_argument('--file_path', default='dataset/', help='file path that contains the dataset')
parser.add_argument('--file_train', default='cnndm_test_top10.csv', help='train data')
parser.add_argument('--file_val', default='cnndm_test_top10.csv', help='val data')
parser.add_argument('--file_test', default='cnndm_test_top10.csv', help='test data')
parser.add_argument('--file_type', default='csv', help='type of file e.g. csv, txt')
parser.add_argument('--pretrained_embed', default='glove.6B.100d', help='which pretrained word embedding to use')
parser.add_argument('--max_vocab_size', type=int, default=50000, help='maximum size of vocabulary')
parser.add_argument('--finetune', type=str2bool, default=False, help='finetune embedding layers?')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clip')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for optimizer')
parser.add_argument('--file_output', default='outputs/test.out', help='output label file')
args = parser.parse_args()
logging.info(args)

ARTICLE = data.Field(sequential=True, include_lengths = True)
SUMMARY = data.Field(sequential=True, include_lengths = True)
QUANTITY_LABEL = data.Field(sequential=True, include_lengths = True, dtype=torch.float, unk_token=None)
# QUANTITY_LABEL_I = data.Field(sequential=True, include_lengths = True, dtype=torch.float, unk_token=None)
# QUANTITY_LABEL_B = data.Field(sequential=True, include_lengths = True, dtype=torch.float, unk_token=None)

data_fields = {
    'article':('article', ARTICLE),
    'quantity_label':('quantity_label', QUANTITY_LABEL),
    'summary':('summary', SUMMARY)}
logging.info('Loading dataset')
train_data, valid_data, test_data = data.TabularDataset.splits(path=args.file_path,
                                                                train=args.file_train,
                                                                validation=args.file_val,
                                                                test=args.file_test,
                                                                format=args.file_type,
                                                                skip_header=False,
                                                                fields=data_fields)
logging.info('Building vocab')
ARTICLE.build_vocab(train_data, vectors=args.pretrained_embed, vectors_cache='../OpenNMT/.vector_cache',
                    max_size=args.max_vocab_size, unk_init=torch.Tensor.normal_)
SUMMARY.build_vocab(train_data, vectors=args.pretrained_embed, vectors_cache='../OpenNMT/.vector_cache',
                    max_size=args.max_vocab_size, unk_init=torch.Tensor.normal_)
QUANTITY_LABEL.build_vocab(train_data)
print(QUANTITY_LABEL.vocab.stoi)
# QUANTITY_LABEL_I.build_vocab(train_data)
# QUANTITY_LABEL_B.build_vocab(train_data)
# print(QUANTITY_LABEL_I.vocab.stoi)
# print(QUANTITY_LABEL_B.vocab.stoi)
# print(LABEL.vocab.stoi) # {'<pad>': 0, 'O': 1, 'BV': 2, 'BU': 3, 'IU': 4, 'IV': 5}
# print(LABEL.vocab.itos) # ['<pad>', 'O', 'BV', 'BU', 'IU', 'IV']
TARGET_TO_IX = {'<pad>': 0, 'O': 1, 'BV': 2, 'BU': 3, 'IU': 4, 'IV': 5}

logging.info(f"Unique tokens in ARTICLE vocabulary: {len(ARTICLE.vocab)}")
logging.info(f"Unique tokens in SUMMARY vocabulary: {len(SUMMARY.vocab)}")

device = torch.device('cuda')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = args.batch_size,
    sort=False,
    shuffle=False,
    device = device)

logging.info('Article embedding '+str(ARTICLE.vocab.vectors.shape[0])+ ', '+str(ARTICLE.vocab.vectors.shape[1]))
logging.info('Summary embedding '+str(SUMMARY.vocab.vectors.shape[0])+ ', '+str(SUMMARY.vocab.vectors.shape[1]))
# logging.info('Label embedding '+str(LABEL.vocab.vectors.shape[0])+ ', '+str(LABEL.vocab.vectors.shape[1]))
pretrained_embeddings_a = ARTICLE.vocab.vectors
pretrained_embeddings_s = SUMMARY.vocab.vectors


encoder_article = Encoder_article_RNN(vocab_size=len(ARTICLE.vocab),
                                      embedding_dim=args.embed_dim,
                                      hidden_dim=args.src_hidden_dim,
                                      n_layers=args.src_num_layers,
                                      bidirectional=args.src_bidirectional,
                                      dropout=args.dropout,
                                      pad_idx=ARTICLE.vocab.stoi[ARTICLE.pad_token])

attention_article = Attention(encoder_hidden_dim=args.src_hidden_dim,
                              decoder_hidden_dim=args.tgt_hidden_dim,
                              attention_dim=args.attention_dim)


decoder_forward = Decoder_LSTM(vocab_size=len(ARTICLE.vocab),
                               embedding_dim=args.embed_dim,
                               encoder_hidden_dim=args.src_hidden_dim,
                               decoder_hidden_dim=args.tgt_hidden_dim,
                               output_dim=len(TARGET_TO_IX),
                               dropout=args.dropout,
                               attention_article=attention_article)

decoder_backward = Decoder_LSTM(vocab_size=len(ARTICLE.vocab),
                               embedding_dim=args.embed_dim,
                               encoder_hidden_dim=args.src_hidden_dim,
                               decoder_hidden_dim=args.tgt_hidden_dim,
                               output_dim=len(TARGET_TO_IX),
                               dropout=args.dropout,
                               attention_article=attention_article)

encoder_article.embedding.weight.data.copy_(pretrained_embeddings_a)
encoder_article.embedding.weight.requires_grad = args.finetune
# encoder_article = nn.DataParallel(encoder_article, device_ids=[0,1])
PAD_IDX = 0
verification_model = Verification_BiLSTM_CRF_allennlp(encoder_article=encoder_article,
                                             decoder_LSTM_f=decoder_forward,
                                             decoder_LSTM_b=decoder_backward,
                                             device=device,
                                             PAD_IDX=SUMMARY.vocab.stoi[SUMMARY.pad_token],)


decoder_forward.embedding.weight.data.copy_(pretrained_embeddings_s)
decoder_forward.embedding.weight.requires_grad = args.finetune
decoder_backward.embedding.weight.data.copy_(pretrained_embeddings_s)
decoder_backward.embedding.weight.requires_grad = args.finetune
logging.info('pre-trained vectors copied to embedding layer')

optimizer = optim.Adam(verification_model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
criterion_binary = nn.BCEWithLogitsLoss()

verification_model = verification_model.to(device)

criterion = criterion.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

c_encoder = count_parameters(encoder_article)
logging.info(f'The article encoder has ' + str(c_encoder) + ' trainable parameters')
c = count_parameters(verification_model)
logging.info(f'The verification model has ' + str(c) + ' trainable parameters')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0
    preds = []

    with torch.no_grad():
        with tqdm(total=len(iterator)) as t:
            with open(args.file_output, 'w') as f_label, open(args.file_output + '.global.scores', 'w') as f_score,\
                open(args.file_output + '.local.scores', 'w') as f_score_local:
                for batch in iterator:
                    article, article_lengths = batch.article

                    summary, summary_lengths = batch.summary

                    quantity_label, _ = batch.quantity_label
                    # quantity_label_I, _ = batch.quantity_label_I
                    # quantity_label_B, _ = batch.quantity_label_B
                    # quantity_label = torch.cat((quantity_label_I.unsqueeze(2), quantity_label_B.unsqueeze(2)), dim=2)

                    output, output_binary, output_raw = model(article, article_lengths, summary, summary_lengths, quantity_label)
                    # _, output, output_binary = model(article, article_lengths, summary, summary_lengths, quantity_label)
                    # output = [batch_size, sentence_len]  It is no longer Tensor, just normal list! also it already exclude PAD
                    # output_raw = [batch_size, seq_len, output_dim]

                    output_binary = output_binary.squeeze(1)  # [batch_size]

                    # preds.append(torch.argmax(output, dim=2).tolist())
                    # pred = torch.argmax(output, dim=2)
                    # raw = torch.max(output, dim=2)[0]
                    # prob = torch.max(torch.softmax(output, dim=2), dim=2)[0] # probabilities
                    # write_preds(pred, raw, prob, summary_lengths)
                    write_labels_and_scores(output, summary_lengths.tolist(), torch.sigmoid(output_binary).tolist(), f_label,
                                            f_score)
                    calculate_and_write_local_score(output, output_raw, summary_lengths.tolist(), f_score_local)
                    t.update()

    return epoch_loss / len(iterator)

def eval_using_checkpoint(model, checkpoint):
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    # model.load_state_dict(ckpt)
    logging.info('checkpoint loaded')
    valid_loss = evaluate(model, test_iterator, criterion)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def write_preds(pred, raw, prob, summary_lengths):
    # pred = [batch_size, sentence_len]
    # sentence_len = pred.shape[0]
    batch_size = len(pred)
    # raw = raw.permute(1,0)
    # raw = raw.tolist()
    prob = prob.permute(1,0)
    log_prob = torch.log(prob).tolist()
    summary_lens = summary_lengths.tolist()
    with open(args.file_output, 'a') as f,\
            open(args.file_output+'.scores.avgUVLogProb', 'a') as ff:
        # label = {'<pad>': 0, 'O': 1, 'BV': 2, 'BU': 3, 'IU': 4, 'IV': 5}
        label = ['<pad>', 'O', 'BV', 'BU', 'IU', 'IV']
        for i in range(batch_size):
            score = 0.
            count_U_V = 0
            for j in range(summary_lens[i]):
                l = label[pred[i][j]]
                f.write(l)
                f.write(' ')
                if l in ['BV','IV']: # if l is BV or IV
                    # score += raw[i][j]
                    score += log_prob[i][j]
                    count_U_V += 1
                elif l in ['BU','IU']: # if l is BU or IU
                    # score -= raw[i][j]
                    score  -= log_prob[i][j]
                    count_U_V += 1

            f.write('\n')
            if count_U_V == 0:
                ff.write(str(0.0))
            else:
                ff.write(str(score/count_U_V))
            ff.write('\n')

def write_labels_and_scores(pred, summary_lengths, output_binary, f_label, f_score):
    # pred = [batch_size, (sentence_len, score)]
    batch_size = len(pred)
    label = ['<pad>', 'O', 'BV', 'BU', 'IV', 'IU']
    for i in range(batch_size):
        for j in range(len(pred[i][0])):
            l = label[pred[i][0][j]]
            f_label.write(l)
            f_label.write(' ')
        f_label.write('\n')
        # write the scores
        f_score.write(str(output_binary[i]))
        f_score.write('\n')

def calculate_and_write_local_score(pred, output, summary_lengths, f_score_local):
    # pred = [batch_size, (sentence_len, score)]
    # output = [batch_size, seq_len, output_dim]
    batch_size = output.shape[0]
    label = ['<pad>', 'O', 'BV', 'BU', 'IV', 'IU']
    label_d = {'<pad>': 0, 'O': 1, 'BV': 2, 'BU': 3, 'IV': 4, 'IU': 5}
    prob = torch.softmax(output, dim=2)
    # print(prob)
    # log_prob = torch.log(prob).tolist()
    # print(log_prob)
    prob = prob.tolist()
    for i in range(batch_size):
        score = 0.
        count_U_V = 0
        for j in range(len(pred[i][0])):
            l = label[pred[i][0][j]]
            if l in ['BV', 'IV']:  # if l is BV or IV
                score += prob[i][j][label_d[l]]
                count_U_V += 1
            elif l in ['BU', 'IU']:  # if l is BU or IU
                score -= prob[i][j][label_d[l]]
                count_U_V += 1

        # write the scores
        if count_U_V == 0:
            f_score_local.write('0.0')
        else:
            f_score_local.write(str(score/count_U_V))
        f_score_local.write('\n')

if __name__ == '__main__':
    mode = args.task
    if mode == 'test':
        logging.info('Starting to test using '+args.checkpoint)
        eval_using_checkpoint(verification_model, args.checkpoint)
    else:
        print('wrong mode')







