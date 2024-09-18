import numpy as np
from scipy.spatial import distance
from shapely.geometry import LineString, CAP_STYLE, JOIN_STYLE, MultiLineString, Polygon
from shapely.strtree import STRtree


def tp_fp_gen(gt, pre, line_width=1, metric='huawei', thr=0.5):
    num_pre = pre.shape[0]
    num_gt = gt.shape[0]
    tp = np.zeros(num_pre, dtype=np.uint8)
    fp = np.zeros(num_pre, dtype=np.uint8)
    if num_gt == 0:
        fp[...] = 1
        return tp, fp, num_pre, num_gt
    if num_pre == 0:
        return tp, fp, num_pre, num_gt
    if metric in ('huawei', ):
        pred_lines_shapely = [Polygon(i).convex_hull for i in pre]
        gt_lines_shapely = [Polygon(i).convex_hull for i in gt]
    else:
        pred_lines_shapely = [LineString(j).buffer(line_width,
                                                   cap_style=CAP_STYLE.flat,
                                                   join_style=JOIN_STYLE.mitre)#线宽是2.0米
                              for j in pre]
        gt_lines_shapely = [LineString(j).buffer(line_width,
                                                 cap_style=CAP_STYLE.flat,
                                                 join_style=JOIN_STYLE.mitre)
                            for j in gt]
    
    

        
        
        
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict((id(pt), j) for j, pt in enumerate(pred_lines_shapely))
    iou_matrix = np.zeros((num_pre, num_gt), dtype=np.float64)
    if metric == 'chamfer':
        iou_matrix = np.full((num_pre, num_gt), -100.)
    for x, pline in enumerate(gt_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]
                if metric == 'chamfer':
                    dist_mat = distance.cdist(pre[pred_id], gt[x], 'euclidean')
                    valid_ab = dist_mat.min(-1).mean()
                    valid_ba = dist_mat.min(-2).mean()
                    iou_matrix[pred_id, x] = -(valid_ba + valid_ab) / 2
                else:
                    inter = o.intersection(pline).area
                    union = o.union(pline).area
                    iou_matrix[pred_id, x] = inter / union
    matrix_max = iou_matrix.max(axis=1)
    matrix_argmax = iou_matrix.argmax(axis=1)
    gt_covered = np.zeros(num_gt, dtype=bool)
    for x in range(num_pre):
        if matrix_max[x] >= thr:
            matched_gt = matrix_argmax[x]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[x] = 1
            else:
                fp[x] = 1
        else:
            fp[x] = 1
    return tp, fp, num_pre, num_gt


def mile_gen(gt, pre):
    num_pre = pre.shape[0]
    num_gt = gt.shape[0]
    # 处理真值
    if num_gt == 0:
        gt_l = 0
    else:
        gt_lines_multi = MultiLineString(gt.tolist())#直接用to list 合并为多条线
        gt_l = gt_lines_multi.length
    if num_pre == 0:
        return 0, [], 0, gt_l
    pred_lines_multi = MultiLineString(pre.tolist())
    if num_gt == 0:
        return 0, [], pred_lines_multi.length, gt_l
    # 计算平均距离误差
    pred_lines_multi_shapely = pred_lines_multi.buffer(0.4,
                                                       cap_style=CAP_STYLE.flat,
                                                       join_style=JOIN_STYLE.mitre)
    distance_list = []
    for sin_gt_line in gt_lines_multi.geoms:
        for sin_pred_line in pred_lines_multi.geoms:
            part = sin_gt_line.intersection(sin_pred_line.buffer(0.4,
                                                                 cap_style=CAP_STYLE.flat,
                                                                 join_style=JOIN_STYLE.mitre))
            if part.length > 0:
                distances = np.linspace(0, part.length, 10)
                points = [part.interpolate(d) for d in distances]
                point_distances = [point.distance(sin_pred_line) for point in points]
                distance_list.append(np.array(point_distances).mean().item())
    # 里程指标
    intersection = gt_lines_multi.intersection(pred_lines_multi_shapely)#预测线buffer 0.4 然后和gt相交
    tp = 0
    if intersection.length > 0:
        tp = intersection.length
    return tp, distance_list, pred_lines_multi.length, gt_l


def up_gen(result, cls_id=0, num_sample=100):
    mask = result['gt_label'] == cls_id
    gt_pts = result['gt_pts'][mask]
    if mask.sum() > 0:
        if gt_pts.shape[1] < num_sample:
            up_pts = []
            for i in gt_pts:
                line = LineString(i)
                distances = np.linspace(0, line.length, num_sample)
                points = np.array([line.interpolate(d).coords for d in distances]).reshape(-1, 2)
                up_pts.append(points)
            gt_pts = np.stack(up_pts)
    mask = result['labels'] == cls_id
    scores = result['scores'][mask]
    pts = result['pts'][mask]
    if mask.sum() > 0:
        if pts.shape[1] < num_sample:
            up_pts = []
            for i in pts:
                line = LineString(i)
                distances = np.linspace(0, line.length, num_sample)
                points = np.array([line.interpolate(d).coords for d in distances]).reshape(-1, 2)
                up_pts.append(points)
            pts = np.stack(up_pts)
    return gt_pts, pts, scores
