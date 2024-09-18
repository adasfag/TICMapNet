from functools import partial
from multiprocessing import Pool

from model.evaluator.gen import up_gen, tp_fp_gen, mile_gen


def up_sample_pool(results, i, num_sample):
    fn = up_gen
    fn = partial(fn, cls_id=i, num_sample=num_sample)
    pool = Pool()
    gen = pool.starmap(fn, zip(results))
    gt_pt, pt, score = tuple(zip(*gen))
    pool.close()
    return gt_pt, pt, score


def get_tp_fp_pool(gt_pt, pre_pt, line_width, metric, thr):
    fn = tp_fp_gen
    fn = partial(fn, line_width=line_width, metric=metric, thr=thr)
    pool = Pool()
    gen = pool.starmap(fn, zip(gt_pt, pre_pt))
    tps, fps, num_pres, num_gts = tuple(zip(*gen))
    pool.close()
    return tps, fps, num_pres, num_gts


def get_mile_pool(gt_pt, pre_pt):
    fn = mile_gen
    pool = Pool()
    gen = pool.starmap(fn, zip(gt_pt, pre_pt))
    tp, distance_list, pre_length, gt_length = tuple(zip(*gen))
    pool.close()
    return tp, distance_list, pre_length, gt_length
