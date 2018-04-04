import argparse

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')

   
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--optim', default='adam',
                                help='optimizer type')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=1,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=16,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=10,
                                help='train epochs')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', nargs='+',
                               default=['/devdata1/zhao/preprocessed/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    path_settings.add_argument('--dev_files', nargs='+',
                               default=['/devdata1/zhao/preprocessed/devset/search.dev.json'],
                               help='list of files that contain the preprocessed dev data')
    path_settings.add_argument('--test_files', nargs='+',
                               default=['/devdata1/zhao/test1set/preprocessed/zhidao.test1.json'],
                               help='list of files that contain the preprocessed test data')
    path_settings.add_argument('--brc_dir', default='../data/baidu',
                               help='the dir with preprocessed baidu reading comprehension data')
    path_settings.add_argument('--vocab_dir', default='../data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='../data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='../data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='../data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    
    parser.add_argument('--name', type=str, default="r-net")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--epoch_num', type=int, default=50)
   
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, default="checkpoint")
    parser.add_argument('--resume', type=str, default='/home/lab713/data1/ipython/zhao/DuReader/pytorch/checkpoint/r-net_Apr-02_23-29/checkpoint.pth.tar')
    #/home/lab713/data1/ipython/zhao/DuReader/pytorch/checkpoint/r-net_Apr-02_23-29/checkpoint.pth.tar
    
    parser.add_argument('--update_word_embedding', type=bool, default=False)
    parser.add_argument('--hidden_size', type=int, default=75)
    parser.add_argument('--attention_size', type=int, default=75)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--residual', type=bool, default=False)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--app_path', type=str, default='/home/lab713/data1/ipython/zhao/DuReader/pytorch/')
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--test_batch_size', type=int, default=1)

    return parser.parse_args()
