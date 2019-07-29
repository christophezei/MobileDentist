import os
from cores.utils.coco_utils import results2json, coco_eval

dir_path = os.path.dirname(os.getcwd())

# output
out_file = dir_path + '/demo/result'

# input
ann_file = dir_path + '/datasets/coco/annotations/instances_val2017.json'


def main():

        # eval bbox
        #  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.257
        #  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.439
        #  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.262
        #  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
        #  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.277
        #  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.240
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.354
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
        #  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.585

        # revised
        #  Average Precision  (AP) @[ IoU=0.50      | maxDets=  1 ] = 0.310
        #  Average Precision  (AP) @[ IoU=0.50      | maxDets= 10 ] = 0.431
        #  Average Precision  (AP) @[ IoU=0.50      | maxDets=100 ] = 0.443
        #  Average Precision  (AP) @[ IoU=0.55      | maxDets=  1 ] = 0.298
        #  Average Precision  (AP) @[ IoU=0.55      | maxDets= 10 ] = 0.411
        #  Average Precision  (AP) @[ IoU=0.55      | maxDets=100 ] = 0.421
        #  Average Precision  (AP) @[ IoU=0.60      | maxDets=  1 ] = 0.281
        #  Average Precision  (AP) @[ IoU=0.60      | maxDets= 10 ] = 0.382
        #  Average Precision  (AP) @[ IoU=0.60      | maxDets=100 ] = 0.390
        #  Average Precision  (AP) @[ IoU=0.65      | maxDets=  1 ] = 0.263
        #  Average Precision  (AP) @[ IoU=0.65      | maxDets= 10 ] = 0.350
        #  Average Precision  (AP) @[ IoU=0.65      | maxDets=100 ] = 0.355
        #  Average Precision  (AP) @[ IoU=0.70      | maxDets=  1 ] = 0.238
        #  Average Precision  (AP) @[ IoU=0.70      | maxDets= 10 ] = 0.308
        #  Average Precision  (AP) @[ IoU=0.70      | maxDets=100 ] = 0.312
        #  Average Precision  (AP) @[ IoU=0.75      | maxDets=  1 ] = 0.206
        #  Average Precision  (AP) @[ IoU=0.75      | maxDets= 10 ] = 0.260
        #  Average Precision  (AP) @[ IoU=0.75      | maxDets=100 ] = 0.262
        #  Average Precision  (AP) @[ IoU=0.80      | maxDets=  1 ] = 0.165
        #  Average Precision  (AP) @[ IoU=0.80      | maxDets= 10 ] = 0.201
        #  Average Precision  (AP) @[ IoU=0.80      | maxDets=100 ] = 0.202
        #  Average Precision  (AP) @[ IoU=0.85      | maxDets=  1 ] = 0.111
        #  Average Precision  (AP) @[ IoU=0.85      | maxDets= 10 ] = 0.130
        #  Average Precision  (AP) @[ IoU=0.85      | maxDets=100 ] = 0.130
        #  Average Precision  (AP) @[ IoU=0.90      | maxDets=  1 ] = 0.048
        #  Average Precision  (AP) @[ IoU=0.90      | maxDets= 10 ] = 0.053
        #  Average Precision  (AP) @[ IoU=0.90      | maxDets=100 ] = 0.053
        #  Average Precision  (AP) @[ IoU=0.95      | maxDets=  1 ] = 0.005
        #  Average Precision  (AP) @[ IoU=0.95      | maxDets= 10 ] = 0.005
        #  Average Precision  (AP) @[ IoU=0.95      | maxDets=100 ] = 0.005
        #  Average Recall     (AR) @[ IoU=0.50      | maxDets=  1 ] = 0.368
        #  Average Recall     (AR) @[ IoU=0.50      | maxDets= 10 ] = 0.582
        #  Average Recall     (AR) @[ IoU=0.50      | maxDets=100 ] = 0.637
        #  Average Recall     (AR) @[ IoU=0.55      | maxDets=  1 ] = 0.354
        #  Average Recall     (AR) @[ IoU=0.55      | maxDets= 10 ] = 0.557
        #  Average Recall     (AR) @[ IoU=0.55      | maxDets=100 ] = 0.606
        #  Average Recall     (AR) @[ IoU=0.60      | maxDets=  1 ] = 0.338
        #  Average Recall     (AR) @[ IoU=0.60      | maxDets= 10 ] = 0.523
        #  Average Recall     (AR) @[ IoU=0.60      | maxDets=100 ] = 0.565
        #  Average Recall     (AR) @[ IoU=0.65      | maxDets=  1 ] = 0.319
        #  Average Recall     (AR) @[ IoU=0.65      | maxDets= 10 ] = 0.478
        #  Average Recall     (AR) @[ IoU=0.65      | maxDets=100 ] = 0.509
        #  Average Recall     (AR) @[ IoU=0.70      | maxDets=  1 ] = 0.291
        #  Average Recall     (AR) @[ IoU=0.70      | maxDets= 10 ] = 0.421
        #  Average Recall     (AR) @[ IoU=0.70      | maxDets=100 ] = 0.443
        #  Average Recall     (AR) @[ IoU=0.75      | maxDets=  1 ] = 0.258
        #  Average Recall     (AR) @[ IoU=0.75      | maxDets= 10 ] = 0.360
        #  Average Recall     (AR) @[ IoU=0.75      | maxDets=100 ] = 0.373
        #  Average Recall     (AR) @[ IoU=0.80      | maxDets=  1 ] = 0.215
        #  Average Recall     (AR) @[ IoU=0.80      | maxDets= 10 ] = 0.287
        #  Average Recall     (AR) @[ IoU=0.80      | maxDets=100 ] = 0.295
        #  Average Recall     (AR) @[ IoU=0.85      | maxDets=  1 ] = 0.158
        #  Average Recall     (AR) @[ IoU=0.85      | maxDets= 10 ] = 0.203
        #  Average Recall     (AR) @[ IoU=0.85      | maxDets=100 ] = 0.207
        #  Average Recall     (AR) @[ IoU=0.90      | maxDets=  1 ] = 0.084
        #  Average Recall     (AR) @[ IoU=0.90      | maxDets= 10 ] = 0.101
        #  Average Recall     (AR) @[ IoU=0.90      | maxDets=100 ] = 0.103
        #  Average Recall     (AR) @[ IoU=0.95      | maxDets=  1 ] = 0.014
        #  Average Recall     (AR) @[ IoU=0.95      | maxDets= 10 ] = 0.017
        #  Average Recall     (AR) @[ IoU=0.95      | maxDets=100 ] = 0.017

        # coco_eval(
        #     result_file=out_file+'.bbox.json',
        #     result_type='bbox',
        #     coco=ann_file,
        #     # iou_thrs=np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True),
        #     iou_thrs=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        #     max_dets=[1, 10, 100]
        # )

        # eval
        # Average Precision  (AP) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.439
        # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = -1.000
        # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = -1.000
        # Average Precision  (AP) @[ IoU=0.50:0.50 | area= small | maxDets=100 ] = 0.161
        # Average Precision  (AP) @[ IoU=0.50:0.50 | area=medium | maxDets=100 ] = 0.503
        # Average Precision  (AP) @[ IoU=0.50:0.50 | area= large | maxDets=100 ] = 0.668
        # Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=  1 ] = 0.365
        # Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets= 10 ] = 0.583
        # Average Recall     (AR) @[ IoU=0.50:0.50 | area=   all | maxDets=100 ] = 0.639
        # Average Recall     (AR) @[ IoU=0.50:0.50 | area= small | maxDets=100 ] = 0.282
        # Average Recall     (AR) @[ IoU=0.50:0.50 | area=medium | maxDets=100 ] = 0.745
        # Average Recall     (AR) @[ IoU=0.50:0.50 | area= large | maxDets=100 ] = 0.881

        # revised
        #  Average Precision  (AP) @[ IoU=0.50      | maxDets=  1 ] = 0.310
        #  Average Precision  (AP) @[ IoU=0.50      | maxDets= 10 ] = 0.431
        #  Average Precision  (AP) @[ IoU=0.50      | maxDets=100 ] = 0.443
        #  Average Precision  (AP) @[ IoU=0.60      | maxDets=  1 ] = 0.281
        #  Average Precision  (AP) @[ IoU=0.60      | maxDets= 10 ] = 0.382
        #  Average Precision  (AP) @[ IoU=0.60      | maxDets=100 ] = 0.390
        #  Average Precision  (AP) @[ IoU=0.70      | maxDets=  1 ] = 0.238
        #  Average Precision  (AP) @[ IoU=0.70      | maxDets= 10 ] = 0.308
        #  Average Precision  (AP) @[ IoU=0.70      | maxDets=100 ] = 0.312
        #  Average Precision  (AP) @[ IoU=0.75      | maxDets=  1 ] = 0.206
        #  Average Precision  (AP) @[ IoU=0.75      | maxDets= 10 ] = 0.260
        #  Average Precision  (AP) @[ IoU=0.75      | maxDets=100 ] = 0.262
        #  Average Precision  (AP) @[ IoU=0.80      | maxDets=  1 ] = 0.165
        #  Average Precision  (AP) @[ IoU=0.80      | maxDets= 10 ] = 0.201
        #  Average Precision  (AP) @[ IoU=0.80      | maxDets=100 ] = 0.202
        #  Average Recall     (AR) @[ IoU=0.50      | maxDets=  1 ] = 0.368
        #  Average Recall     (AR) @[ IoU=0.50      | maxDets= 10 ] = 0.582
        #  Average Recall     (AR) @[ IoU=0.50      | maxDets=100 ] = 0.637
        #  Average Recall     (AR) @[ IoU=0.60      | maxDets=  1 ] = 0.338
        #  Average Recall     (AR) @[ IoU=0.60      | maxDets= 10 ] = 0.523
        #  Average Recall     (AR) @[ IoU=0.60      | maxDets=100 ] = 0.565
        #  Average Recall     (AR) @[ IoU=0.70      | maxDets=  1 ] = 0.291
        #  Average Recall     (AR) @[ IoU=0.70      | maxDets= 10 ] = 0.421
        #  Average Recall     (AR) @[ IoU=0.70      | maxDets=100 ] = 0.443
        #  Average Recall     (AR) @[ IoU=0.75      | maxDets=  1 ] = 0.258
        #  Average Recall     (AR) @[ IoU=0.75      | maxDets= 10 ] = 0.360
        #  Average Recall     (AR) @[ IoU=0.75      | maxDets=100 ] = 0.373
        #  Average Recall     (AR) @[ IoU=0.80      | maxDets=  1 ] = 0.215
        #  Average Recall     (AR) @[ IoU=0.80      | maxDets= 10 ] = 0.287
        #  Average Recall     (AR) @[ IoU=0.80      | maxDets=100 ] = 0.295

        coco_eval(
            result_file=out_file+'.bbox.json',
            result_type='bbox',
            coco=ann_file,
            iou_thrs=[0.2, 0.3, 0.4, 0.5],
            max_dets=[10]
        )


if __name__ == '__main__':
    main()
