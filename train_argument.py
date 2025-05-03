import argparse

def parser():
   #This creates the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=["Base_CNN"], default="Base_CNN", #This adds an argument to the parser
        help='Which model to use') #These arguments can be access with args.name, where args = parser.parse_args()
    parser.add_argument('--dataset', choices=['mnist'], default ='mnist')
    parser.add_argument('--data_root', default='data', 
        help='the directory to save the dataset')
    parser.add_argument('--log_root', default='log', 
        help='the directory to save the logs or other imformations (e.g. images)')
    parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
    parser.add_argument('--load_checkpoint', default='./model/default/model.pth')
    parser.add_argument('--affix', default='natural_train', help='the affix for the save folder')
    

    ## Training realted 
    parser.add_argument('--num_clients', '-N', type=int, default=50, help='number of clients')
    parser.add_argument('--n_bit', type = int, default = 16, help = 'quantization level for local training')
    parser.add_argument('--m_bit', type = int, default = 16, help = 'quantization level for transmission')
    parser.add_argument('--schedulingsize', type=int, default = 5, help = 'how many clients will be sampled')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--comm_rounds', '-m_e', type=int, default=20, 
        help='the maximum communication rounds')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum(defalt: 0.9)")
    parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
    parser.add_argument('--seed', default=1, help='The random seed')
    parser.add_argument('--alpha', type=float, default=0.1, help="Dirichelet concentration parameter")
    parser.add_argument('--weight_decay', type=float, default=0., help="SGD weight decay(defalt: 0.)")
    parser.add_argument('--local_epoch', type=int, default = 5, help = "number of local iterations (default = 5)")

    # Qmix related
    parser.add_argument('--use_qmix', type=bool, default=True,  # 默认启用QMIX
                   help='是否使用QMIX算法进行客户端选择和量化预算分配')
    parser.add_argument('--qmix_model_path', type=str, default='', help='QMIX模型加载路径')
    parser.add_argument('--sample_ratio', type=float, default=0.1, 
                        help='客户端采样率(默认使用schedulingsize/num_clients)')
    parser.add_argument('--rnn_hidden_dim', type=int, default=64, help='QMIX RNN隐藏层维度')
    parser.add_argument('--qmix_epsilon', type=float, default=0.5, help='QMIX初始探索率')
    parser.add_argument('--qmix_epsilon_decay', type=float, default=0.995, help='QMIX探索率衰减')
    parser.add_argument('--qmix_epsilon_min', type=float, default=0.05, help='QMIX最小探索率')
    parser.add_argument('--energy_weight', type=float, default=0.01, help='能耗惩罚权重')
    parser.add_argument('--time_weight', type=float, default=0.005, help='时间惩罚权重')

    # 性能比较
    parser.add_argument('--aggregation_method', type=str, choices=['original', 'weighted', 'compare'], 
                    default='compare', help='选择聚合方法: original=原始均等权重, weighted=基于量化精度的加权, compare=两种方法同时运行并比较')
    parser.add_argument('--log_aggregation_weights', type=lambda x: str(x).lower() == 'true', 
                    default=True, help='是否记录聚合权重到日志')
    parser.add_argument('--save_comparison_plot', type=lambda x: str(x).lower() == 'true',
                    default=True, help='是否保存比较结果图表')
    parser.add_argument('--comparison_mode', type=str, choices=['none', 'aggregation', 'comprehensive'], 
                default='comprehensive', help='比较模式: none=无比较模式, aggregation=仅比较聚合方法, comprehensive=全面比较QMIX和聚合方法')

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
