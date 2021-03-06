dict(
    # lrate decay
    # select strategy: noam, gnmt+, epoch, score and vanilla
    lrate_strategy="epoch",
    # learning decay rate
    lrate_decay=0.5,
    # weight decay for L2 loss
    weight_decay=3e-5,

    # early stopping
    estop_patience=100,

    # initialization
    # type of initializer
    initializer="uniform",
    # initializer range control
    initializer_gain=0.08,

    # parameters for rnnsearch
    # encoder and decoder hidden size
    hidden_size=300,
    # source and target embedding size
    embed_size=300,
    # label number
    label_size=3,
    # number of layers
    char_embed_size=64,
    # dropout value
    dropout=0.3,
    # label smoothing value
    label_smooth=0.1,
    # gru, lstm, sru or atr
    cell="atr",
    # whether use layer normalization, it will be slow
    layer_norm=False,
    # notice that when opening the swap memory switch
    # you can train reasonably larger batch on condition
    # that your system will use much more cpu memory
    swap_memory=True,

    # bert configuration
    bert=None,
    bert_dir="path-to-bert/cased_L-12_H-768_A-12",
    tune_bert=False,
    enable_bert=False,
    use_bert_single=True,

    # whether use character embedding
    use_char=True,
    # whether lowercase word
    lower=False,
    bert_lower=False,

    model_name="nlinet",

    # constant batch size at 'batch' mode for batch-based batching
    batch_size=128,
    token_size=2000,
    batch_or_token='batch',
    # batch size for decoding, i.e. number of source sentences decoded at the same time
    eval_batch_size=64,
    # whether shuffle batches during training
    shuffle_batch=True,
    # whether use multiprocessing deal with data reading, default true
    data_multiprocessing=True,

    # word vocabulary
    word_vocab_file="path-of/word_vocab",
    # char vocabulary
    char_vocab_file="path-of/char_vocab",
    # pretrained word embedding
    pretrain_word_embedding_file="path-of/word_vocab.npz",
    # train file
    train_file=["path-of/train.p", "path-of/train.q", "path-of/train.l"],
    # dev file
    dev_file=["path-of/dev.p", "path-of/dev.q", "path-of/dev.l"],
    # test file
    test_file=["path-of/test.p", "path-of/test.q", "path-of/test.l"],
    # output directory
    output_dir="train",
    # output during testing
    test_output="",

    # adam optimizer hyperparameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    # gradient clipping value
    clip_grad_norm=5.0,
    # initial learning rate
    lrate=1e-3,

    # allowed maximum sentence length
    max_len=100,
    # maximum word length
    max_w_len=25,

    # maximum epochs
    epoches=10,
    # the effective batch size is: batch/token size * update_cycle
    # sequential update cycle
    update_cycle=1,
    # the number of gpus
    gpus=[0],
    # whether enable ema
    ema_decay=0.9999,

    # print information every disp_freq training steps
    disp_freq=10,
    # evaluate on the development file every eval_freq steps
    eval_freq=1000,
    # save the model parameters every save_freq steps
    save_freq=1000,
    # saved checkpoint number
    checkpoints=5,
    # the maximum training steps, program with stop if epoches or max_training_steps is metted
    max_training_steps=100000,

    # number of threads for threaded reading, seems useless
    nthreads=6,
    # buffer size controls the number of sentences readed in one time,
    buffer_size=20000,
    # a unique queue in multi-thread reading process
    max_queue_size=100,
    # random control, not so well for tensorflow.
    random_seed=1234,
    # whether or not train from checkpoint
    train_continue=True,
)
