# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import json
import tensorflow as tf

from utils import queuer


def decoding(sprobs, samples, params, mask=None):
    """Generate decoded sequence from seqs"""
    if mask is None:
        mask = [1.] * len(sprobs)

    flat_sprobs = []
    for _sprobs, _m in zip(sprobs, mask):
        if _m < 1.: continue

        for start_prob in _sprobs:
            flat_sprobs.append(start_prob)

    assert len(flat_sprobs) == len(samples), 'Decoding length mismatch!'

    results = []

    for (idx, sample), pred in zip(samples, flat_sprobs):
        gold_label = sample[0]
        pred_label = pred

        results.append({
            'pred_answer': int(pred_label),
            'sample_id': idx,
            'gold_answer': gold_label
        })

    return results


def predict(session, features,
            out_pred, dataset, params, train=True):
    """Performing decoding with exising information"""
    results = []

    batcher = dataset.batcher(params.eval_batch_size,
                              buffer_size=params.buffer_size,
                              shuffle=False, train=train)
    eval_queue = queuer.EnQueuer(batcher,
                                 multiprocessing=params.data_multiprocessing,
                                 random_seed=params.random_seed)
    eval_queue.start(workers=params.nthreads,
                     max_queue_size=params.max_queue_size)

    def _predict_one_batch(data_on_gpu):
        feed_dicts = {}
        flat_raw_data = []
        for fidx, data in enumerate(data_on_gpu):
            # define feed_dict
            feed_dict = {
                features[fidx]["p"]: data['p_token_ids'],
                features[fidx]["h"]: data['h_token_ids'],
                features[fidx]["l"]: data['l_id'],
            }
            if params.use_char:
                feed_dict[features[fidx]["pc"]] = data['p_char_ids']
                feed_dict[features[fidx]["hc"]] = data['h_char_ids']

            if params.enable_bert:
                feed_dict[features[fidx]["ps"]] = data['p_subword_ids']
                feed_dict[features[fidx]["hs"]] = data['h_subword_ids']
                feed_dict[features[fidx]["pb"]] = data['p_subword_back']
                feed_dict[features[fidx]["hb"]] = data['h_subword_back']

            feed_dicts.update(feed_dict)
            flat_raw_data.extend(data['raw'])

        # pick up valid outputs
        data_size = len(data_on_gpu)
        valid_out_pred = out_pred[:data_size]

        decode_spred = session.run(
            valid_out_pred, feed_dict=feed_dicts)

        predictions = decoding(
            decode_spred, flat_raw_data, params
        )

        return predictions

    very_begin_time = time.time()
    data_on_gpu = []
    for bidx, data in enumerate(eval_queue.get()):

        data_on_gpu.append(data)
        # use multiple gpus, and data samples is not enough
        if len(params.gpus) > 0 and len(data_on_gpu) < len(params.gpus):
            continue

        start_time = time.time()
        predictions = _predict_one_batch(data_on_gpu)
        data_on_gpu = []
        results.extend(predictions)

        tf.logging.info(
            "Decoding Batch {} using {:.3f} s, translating {} "
            "sentences using {:.3f} s in total".format(
                bidx, time.time() - start_time,
                len(results), time.time() - very_begin_time
            )
        )

    eval_queue.stop()

    if len(data_on_gpu) > 0:
        start_time = time.time()
        predictions = _predict_one_batch(data_on_gpu)
        results.extend(predictions)

        tf.logging.info(
            "Decoding Batch {} using {:.3f} s, translating {} "
            "sentences using {:.3f} s in total".format(
                'final', time.time() - start_time,
                len(results), time.time() - very_begin_time
            )
        )

    return results


def eval_metric(results, params):
    """BLEU Evaluate """

    crr_cnt, total_cnt = 0, 0

    for result in results:
        total_cnt += 1

        p = result['pred_answer']
        g = result['gold_answer']

        if p == g:
            crr_cnt += 1

    return crr_cnt * 100. / total_cnt


def dump_predictions(results, output):
    """save translation"""
    with tf.gfile.Open(output, 'w') as writer:
        for sample in results:
            writer.write(json.dumps(sample) + "\n")
    tf.logging.info("Saving translations into {}".format(output))
