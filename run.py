
from torchtext import data
import torch.optim as optim
import time
import logging
from tqdm import tqdm
import argparse
from models import *
import math
from sklearn.metrics import f1_score, classification_report, confusion_matrix

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
parser.add_argument('--task', default='train', help='train | eval | test')
parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs.')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--checkpoint', default='checkpoint', help='name of saved checkpoint')
parser.add_argument('--input_dim', type=int, default=500002, help='dimension of input')
parser.add_argument('--batch_size', type=int, default=32, help='batch size.')
parser.add_argument('--embed_dim', type=int, default=100, help='embedding dimension')
parser.add_argument('--src_hidden_dim', type=int, default=256, help='encoder hidden dimension')
parser.add_argument('--tgt_hidden_dim', type=int, default=256, help='decoder hidden dimension')
# parser.add_argument('--output_dim', type=int, default=5, help='output dimension')
parser.add_argument('--attention_dim', type=int, default=256, help='attention dimension')
parser.add_argument('--src_num_layers', type=int, default=2, help='encoder number layers')
parser.add_argument('--src_bidirectional', type=str2bool, default=True, help='encoder bidirectional?')
parser.add_argument('--file_path', default='dataset/', help='file path that contains the dataset')
parser.add_argument('--file_train', default='train.csv', help='train data')
parser.add_argument('--file_val', default='val.csv', help='val data')
parser.add_argument('--file_test', default='test.csv', help='test data')
parser.add_argument('--file_type', default='csv', help='type of file e.g. csv, txt')
parser.add_argument('--pretrained_embed', default='glove.6B.100d', help='which pretrained word embedding to use')
parser.add_argument('--max_vocab_size', type=int, default=50000, help='maximum size of vocabulary')
parser.add_argument('--finetune', type=str2bool, default=True, help='finetune embedding layers?')
parser.add_argument('--clip', type=float, default=1.0, help='gradient clip')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for optimizer')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='probability to use teacher forcing')
parser.add_argument('--file_output', default='outputs/test.out', help='output label file')
parser.add_argument('--label_loss_weight', type=float, default=1.0, help='the weight of loss for labels')
parser.add_argument('--label_binary_loss_weight', type=float, default=1.0, help='the weight of loss for the binary labels')
args = parser.parse_args()
logging.info(args)

ARTICLE = data.Field(sequential=True, include_lengths = True)
SUMMARY = data.Field(sequential=True, include_lengths = True)
LABEL = data.Field(sequential=True, include_lengths = True, unk_token=None)
QUANTITY_LABEL = data.Field(sequential=True, include_lengths = True, dtype=torch.float, unk_token=None)
LABEL_BINARY = data.LabelField(sequential=False, dtype=torch.float, use_vocab=False)

data_fields = {
    'label':('label', LABEL),
    'quantity_label':('quantity_label', QUANTITY_LABEL),
    'label_binary':('label_binary', LABEL_BINARY),
    'article':('article', ARTICLE),
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
# vectors = Vectors(name='label.txt', cache='../OpenNMT/.vector_cache')
# LABEL.build_vocab(train_data, vectors=vectors) # vector is 5 dimension # maybe no need to use embedding
LABEL.build_vocab(train_data)
QUANTITY_LABEL.build_vocab(train_data)
TARGET_TO_IX = LABEL.vocab.stoi
print(TARGET_TO_IX)
# print(LABEL.vocab.stoi) # {'<pad>': 0, 'O': 1, 'BV': 2, 'BU': 3, 'IU': 4, 'IV': 5}
print(LABEL.vocab.itos) # ['<pad>', 'O', 'BV', 'BU', 'IU', 'IV'] or ['<pad>', 'O', 'V', 'U']
LABEL_BINARY.build_vocab(train_data)
print(LABEL_BINARY.vocab.stoi)
print(QUANTITY_LABEL.vocab.stoi)
logging.info(f"Unique tokens in ARTICLE vocabulary: {len(ARTICLE.vocab)}")
logging.info(f"Unique tokens in SUMMARY vocabulary: {len(SUMMARY.vocab)}")
logging.info(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")
logging.info(f"Unique tokens in LABEL_BINARY vocabulary: {len(LABEL_BINARY.vocab)}")

device = torch.device('cuda')

if args.task == 'test_xsljsdl':
    # during testing, we don't want to change the order of the testset
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=args.batch_size,
        sort=False,
        shuffle=False,
        device=device)
else:
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = args.batch_size,
        sort_key=lambda x: len(x.article),
        sort_within_batch = True,
        shuffle=True,
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
                               output_dim=len(LABEL.vocab),
                               dropout=args.dropout,
                               attention_article=attention_article)

decoder_backward = Decoder_LSTM(vocab_size=len(ARTICLE.vocab),
                               embedding_dim=args.embed_dim,
                               encoder_hidden_dim=args.src_hidden_dim,
                               decoder_hidden_dim=args.tgt_hidden_dim,
                               output_dim=len(LABEL.vocab),
                               dropout=args.dropout,
                               attention_article=attention_article)

encoder_article.embedding.weight.data.copy_(pretrained_embeddings_a)
encoder_article.embedding.weight.requires_grad = args.finetune
# encoder_article = nn.DataParallel(encoder_article, device_ids=[0,1])
PAD_IDX = LABEL.vocab.stoi['<pad>']
verification_model = Verification_BiLSTM_CRF_allennlp(encoder_article=encoder_article,
                                             decoder_LSTM_f=decoder_forward,
                                             decoder_LSTM_b=decoder_backward,
                                             device=device,
                                             PAD_IDX=SUMMARY.vocab.stoi[SUMMARY.pad_token])


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
criterion_binary = criterion_binary.to(device)

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_parameters(model):
    return sum(p.numel() for p in model.parameters())

c_encoder = count_trainable_parameters(encoder_article)
logging.info(f'The article encoder has ' + str(c_encoder) + ' trainable parameters')
c = count_trainable_parameters(verification_model)
logging.info(f'The verification model has ' + str(c) + ' trainable parameters')
c = count_total_parameters(verification_model)
logging.info(f'The verification model has ' + str(c) + ' parameters in total')

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def f1(preds, y):
    predictions = torch.argmax(preds, dim=1)
    f1 = f1_score(y_pred=predictions.tolist(), y_true=y.tolist(), average='micro')
    return f1

def classification_stats(preds, y):
    print(classification_report(y_pred=preds, y_true=y, labels=list(LABEL.vocab.stoi.values()), target_names=LABEL.vocab.itos, digits=4))

def confusion_matrix_stats(preds, y):
    print(confusion_matrix(y_true=y, y_pred=preds))

def precision_recall(preds, y):
    # please note, this only calculate p, r for one batch
    # in order to calculate for whole dataset, need to return TP, etc, not p, r
    rounded_preds = torch.round(torch.sigmoid(preds))

    TP = ((rounded_preds == 1.) & (y == 1.)).float().sum()
    TN = ((rounded_preds == 0.) & (y == 0.)).float().sum()
    FN = ((rounded_preds == 0.) & (y == 1.)).float().sum()
    FP = ((rounded_preds == 1.) & (y == 0.)).float().sum()
    # p = TP / (TP + FP + 1e-10)
    # r = TP / (TP + FN + 1e-10)
    # F1 = 2 * r * p / (r + p)
    # acc = (TP + TN) / (TP + TN + FP + FN )

    return TP, FP, FN, TN

def train(model, iterator, optimizer, criterion, criterion_binary):

    model.train()
    epoch_loss = 0
    # epoch_acc = 0

    with tqdm(total=len(iterator)) as t:
        for batch in iterator:
            model.zero_grad() # should be equalvalent with optimizer.zero_grad()

            article, article_lengths = batch.article

            summary, summary_lengths = batch.summary

            label, _ = batch.label
            # label = [sent len, batch size]

            quantity_label, _ = batch.quantity_label
            # [seq_len, batch_size, 1]

            output_loss, output_binary = model.neg_log_likelihood(article, article_lengths, summary, summary_lengths, label, quantity_label)

            # output, output_binary = model(article, article_lengths, summary, summary_lengths)
            # output_binary = [batch_size, 1]
            output_binary = output_binary.squeeze(1) # [batch_size]

            loss = args.label_loss_weight * output_loss + args.label_binary_loss_weight * criterion_binary(output_binary, batch.label_binary)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            epoch_loss += float(loss)
            t.update()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, criterion_binary):

    model.eval()

    epoch_loss = 0
    epoch_TP = 0
    epoch_FP = 0
    epoch_FN = 0
    epoch_TN = 0
    preds = []
    labels = []

    with torch.no_grad():
        # with tqdm(total=len(iterator)) as t:
        for batch in iterator:
            article, article_lengths = batch.article

            summary, summary_lengths = batch.summary

            label, _ = batch.label

            quantity_label, _ = batch.quantity_label
            # [seq_len, batch_size, 1]

            output, output_binary, _ = model(article, article_lengths, summary, summary_lengths, quantity_label)
            # output = [batch_size, (sentence_len, score)]  It is no longer Tensor, just normal list! also it already exclude PAD
            # output_binary = [batch_size, 1]

            # we want to truncate because we don't care about PAD token's F1
            label_truncated = truncate_label(label, summary_lengths.tolist())
            # [batch_size, sentence_len]

            output_flatten = []
            for tags, _ in output:
                output_flatten.extend(tags)

            output_binary = output_binary.squeeze(1)  # [batch_size]

            label_truncated = label_truncated.view(-1)

            preds += output_flatten
            labels += label_truncated.tolist()
            # sanity check
            assert len(output_flatten) == label_truncated.shape[0]

            # loss = args.label_loss_weight * output_loss + args.label_binary_loss_weight * criterion_binary(output_binary, batch.label_binary)
            loss = args.label_binary_loss_weight * criterion_binary(output_binary, batch.label_binary)
            epoch_loss += loss.item()

            TP, FP, FN, TN = precision_recall(output_binary, batch.label_binary)
            epoch_TP += TP.item()
            epoch_FP += FP.item()
            epoch_FN += FN.item()
            epoch_TN += TN.item()

        classification_stats(preds, labels)
        confusion_matrix_stats(preds, labels)
        epoch_prec = epoch_TP / (epoch_TP + epoch_FP + 1e-10)
        epoch_rec = epoch_TP / (epoch_TP + epoch_FN + 1e-10)
        epoch_acc = (epoch_TP + epoch_TN) / (epoch_TP + epoch_TN + epoch_FP + epoch_FN + 1e-10)
        epoch_f1 = 2 * epoch_prec * epoch_rec / (epoch_prec + epoch_rec + 1e-10)

    return epoch_loss / len(iterator), epoch_prec, epoch_rec, epoch_f1, epoch_acc

def truncate_label(label, lengths):
    label_truncated = label.permute(1,0)[0,:lengths[0]] # [1, sentence_len]
    for i in range(1, label.shape[1]):
        label_truncated = torch.cat((label_truncated, label.permute(1,0)[i,:lengths[i]]), dim = 0)

    return label_truncated # [batch_size, sentence_len]

def write_labels_and_scores(pred, summary_lengths, output_binary, f_label, f_score):
    # pred = [sentence_len, batch_size]
    batch_size = pred.shape[1]
    pred = pred.permute(1, 0)
    pred = pred.tolist()
    label = ['<pad>', 'O', 'BV', 'BU', 'IU', 'IV']
    for i in range(batch_size):
        for j in range(summary_lengths[i]):
            l = label[pred[i][j]]
            f_label.write(l)
            f_label.write(' ')
        f_label.write('\n')
        # write the scores
        f_score.write(str(output_binary[i]))
        f_score.write('\n')


def eval_using_checkpoint(model, checkpoint):
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    # model.load_state_dict(torch.load(checkpoint))
    logging.info('checkpoint loaded')
    valid_loss, valid_prec, valid_rec, valid_f1, valid_acc = evaluate(model, valid_iterator, criterion, criterion_binary)
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f'\t Val. Acc: {valid_acc*100:.2f}% | Val. Precision: {valid_prec*100:.2f}% | Val. Recall: {valid_rec*100:.2f}% | Val. F1: {valid_f1*100:.2f}%')


def test_using_checkpoint(model, checkpoint):
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    logging.info('checkpoint loaded')
    test_loss, test_prec, test_rec, test_f1, test_acc = evaluate(model, test_iterator, criterion, criterion_binary)
    print(f'\t Test. Loss: {test_loss:.3f} |  Test. PPL: {math.exp(test_loss):7.3f}')
    print(f'\t Test. Acc: {test_acc*100:.2f}% | Val. Precision: {test_prec*100:.2f}% | Val. Recall: {test_rec*100:.2f}% | Val. F1: {test_f1*100:.2f}%')


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def run(model, num_epochs, checkpoint):
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, criterion_binary)
        valid_loss, valid_prec, valid_rec, valid_f1, valid_acc = evaluate(model, valid_iterator, criterion, criterion_binary)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint_dict = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint_dict, './checkpoints/' + checkpoint +'.pt')
            logging.info('checkpoint saved')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print(f'\t Val. Acc: {valid_acc*100:.2f}% | Val. Precision: {valid_prec*100:.2f}% | Val. Recall: {valid_rec*100:.2f}% | Val. F1: {valid_f1*100:.2f}%')


if __name__ == '__main__':

    mode = args.task
    if mode == 'train':
        logging.info('Starting to train')
        run(verification_model, args.num_epoch, args.checkpoint)
    elif mode == 'eval':
        logging.info('Starting to eval using '+args.checkpoint)
        eval_using_checkpoint(verification_model, args.checkpoint)
    elif mode == 'test':
        logging.info('Starting to test using '+args.checkpoint)
        test_using_checkpoint(verification_model, args.checkpoint)
    else:
        print('wrong mode')







