# Copyright (c) Facebook, Inc. and its affiliates.
import io
import copy 
import itertools
import time 
import json
import logging
import shutil
import os
import tempfile
import numpy as np 
from tabulate import tabulate
from collections import OrderedDict
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.evaluation.panoptic_evaluation import COCOPanopticEvaluator
import multiprocessing
from detectron2.data import MetadataCatalog
from panopticapi.evaluation import PQStat
from panopticapi.utils import get_traceback, rgb2id, id2rgb
from imantics import Image as imantics_Image
from imantics import Mask as imantics_Mask
from imantics import Category as imantics_Category
from PIL import Image


def _print_panoptic_results_own(pq_res):
    headers = ["", "PQ", "SQ", "RQ", "#categories"]
    data = []
    for name in ["All", "Things", "Stuff"]:
        row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
        data.append(row)
    table = tabulate(
        data, headers=headers, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center"
    )
    return table 

logger = logging.getLogger(__name__)
OFFSET = 256 * 256 * 256
VOID = 255


class PQStat_own(PQStat):
    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            label = int(label)
            if isthing is not None:
                cat_isthing = int(label_info['isthing']) == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


# proc_id = 0
# annotation_set = annotations_split[0]
@get_traceback
def pq_compute_single_core_own(proc_id, annotation_set, gt_folder, pred_folder, categories):
    pq_stat = PQStat_own()

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        if idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        idx += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
        unique_colors = np.unique(pan_pred.reshape(-1, pan_pred.shape[2]), axis=0).tolist()
        if len(unique_colors) == 1:
            true_id_idx = np.argmax(unique_colors[0])
            pan_pred = copy.deepcopy(pan_pred[:,:,true_id_idx])
            pan_pred = np.stack((pan_pred,)*3, axis=-1)
        pan_pred = rgb2id(pan_pred)

        gt_segms = {float(el['id']): el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}
        
        # predicted segments area calculation + prediction sanity checks
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)
        for label, label_cnt in zip(labels, labels_cnt):
            if label not in pred_segms:
                if label == VOID:
                    continue
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if str(pred_segms[label]['category_id']) not in [str(x) for x in list(categories.keys())]:
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in [float(x) for x in list(gt_segms.keys())]:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if int(gt_segms[gt_label]['category_id']) != int(pred_segms[pred_label]['category_id']):
                continue
            
            union = pred_segms[pred_label]['area'] + int(gt_segms[gt_label]['area']) - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou > 0.50:
                pq_stat[int(gt_segms[gt_label]['category_id'])].tp += 1
                pq_stat[int(gt_segms[gt_label]['category_id'])].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if int(gt_info['iscrowd']) == 1:
                crowd_labels_dict[int(gt_info['category_id'])] = int(gt_label)
                continue
            pq_stat[int(gt_info['category_id'])].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
    print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat



# Copy of pq_compute_multi_core used for computing the PQ metric. Primarily copied for debugging reasons.
def pq_compute_multi_core_own(matched_annotations_list, gt_folder, pred_folder, categories):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core_own,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories))
        processes.append(p)
    pq_stat = PQStat_own()
    for p in processes:
        pq_stat += p.get()
    return pq_stat


# Copy of the pq_compute function used for computing the PQ metric. This is mostly copied into my own code for debugging reasons ....
def pq_compute_own(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):

    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)

    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print("Evaluation panoptic segmentation metrics:")
    print("Ground truth:")
    print("\tSegmentation folder: {}".format(gt_folder))
    print("\tJSON file: {}".format(gt_json_file))
    print("Prediction:")
    print("\tSegmentation folder: {}".format(pred_folder))
    print("\tJSON file: {}".format(pred_json_file))

    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            raise Exception('no prediction for the image with id: {}'.format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    # matched_annotations_list=matched_annotations_list
    # gt_folder=gt_folder
    # pred_folder=pred_folder
    # categories=categories
    pq_stat = pq_compute_multi_core_own(matched_annotations_list=matched_annotations_list, gt_folder=gt_folder, pred_folder=pred_folder, categories=categories)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (10 + 7 * 4))

    for name, _isthing in metrics:
        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n'])
        )

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))

    return results


# Define a custom panoptic evaluator class using the COCO structure of the dataset. This is primarily to save the json files differently 
class Custom_Panoptic_Evaluator(COCOPanopticEvaluator):
    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            ### The panoptic prediction is a tuple of (panoptic_img, panoptic_segments_info). If segments_info is None (or an empty list), create it by using the imantics library  
            if isinstance(segments_info, list):
                if len(segments_info) == 0:
                    segments_info = None 
            if segments_info is None:
                segments_info = list()
                panoptic_img = np.add(panoptic_img, 1)
                panoptic_values = np.unique(panoptic_img)
                for panoptic_value in panoptic_values:
                    binary_mask = (panoptic_img == panoptic_value)
                    # panoptic_value += 1
                    image_raw_data = imantics_Image.from_path(path=input["file_name"])
                    mask_data = imantics_Mask(binary_mask)
                    image_raw_data.add(mask_data, category=imantics_Category("Category_name"))
                    coco_json_panoptic_pred = image_raw_data.export(style="coco")
                    polygon_coordinates = coco_json_panoptic_pred["annotations"][0]["segmentation"]
                    bbox = [float(val) for val in coco_json_panoptic_pred["annotations"][0]["bbox"]]
                    if "vitrolife" in input["file_name"].lower():
                        meta_data = MetadataCatalog.get("vitrolife_dataset_train")
                    else:
                        meta_data = MetadataCatalog.get("ade20k_panoptic_train")
                    assert "vitrolife" in input["file_name"].lower(), "Only vitrolife dataset is supported at the moment!!"
                    panoptic_id_cont_id = meta_data.panoptic_dataset_id_to_contiguous_id
                    category_id_object = np.min([panoptic_value, np.max(list(panoptic_id_cont_id.values()))])
                    object_id = rgb2id(np.stack((panoptic_value,)*3, axis=-1))
                    isthing = "PN" in meta_data.panoptic_classes[category_id_object]
                    segments_info.append({  "id": int(object_id),
                                            "category_id": int(category_id_object),
                                            "isthing": bool(isthing),
                                            "bbox": bbox,
                                            "area": int(np.sum(binary_mask)),
                                            "iscrowd": int(0),
                                            "segmentation": polygon_coordinates,
                                            "image_id": input["image_id"]})
                
            file_name = os.path.basename(input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                self._predictions.append({  "image_id": input["image_id"],
                                            "file_name": file_name_png,
                                            "png_string": out.getvalue(),
                                            "segments_info": segments_info})


    # self = evaluators[Segment_type]
    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        # with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
        pred_dir = tempfile.mkdtemp()
        for p in self._predictions:
            pass 
            with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                p["png_string"]
                f.write(p.pop("png_string"))

        with open(gt_json, "r") as f:
            json_data = json.load(f)
        json_data["annotations"] = self._predictions

        output_dir = self._output_dir or pred_dir
        predictions_json = os.path.join(output_dir, "predictions.json")
        with PathManager.open(predictions_json, "w") as f:
            f.write(json.dumps(json_data, sort_keys=True, indent=4))

        # gt_json_file=gt_json
        # pred_json_file=PathManager.get_local_path(predictions_json)
        # gt_folder=gt_folder
        # pred_folder=pred_dir
        pq_res = pq_compute_own(gt_json_file=gt_json, pred_json_file=PathManager.get_local_path(predictions_json), gt_folder=gt_folder, pred_folder=pred_dir)

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        table = _print_panoptic_results_own(pq_res)

        shutil.rmtree(pred_dir)
        print(table)
        return results


