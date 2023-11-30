import os
import argparse
import gradio as gr
import numpy as np
import torch
from sam import SAM, draw_mask, draw_label_masks
from yolov8 import YOLOv8
from groundingdino import GroundingDINO
from PIL import Image, ImageDraw
from utils.tools import box_prompt, format_results, point_prompt
from utils.tools_gradio import fast_process
import cv2
import time

# Most of our demo code is from [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM). Huge thanks for AN-619.

parser = argparse.ArgumentParser(prog=__file__)
parser.add_argument('--input', type=str, default='./test_imgs', help='path of input')
parser.add_argument('--det_method', type=str, default='groundingdino', help='method for detection') # 当前只支持default
parser.add_argument('--seg_method', type=str, default='mobilesam', help='method for segmentaton') # 当前只支持default
parser.add_argument('--dev_id', type=int, default=0, help='dev id')
parser.add_argument('--conf_thresh', type=float, default=0.25, help='det confidence threshold')
parser.add_argument('--nms_thresh', type=float, default=0.7, help='det nms threshold')
parser.add_argument('--single_output', action='store_true', help='det confidence threshold')
args = parser.parse_args()



if args.det_method == 'yolov8s':
    args.bmodel = './yolov8/BM1684X_models/yolov8s_fp16_1b.bmodel'
    class_names = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush')
    det_pipeline = YOLOv8(args)
elif args.det_method == 'groundingdino':
    args.bmodel = './groundingdino/bmodels/groundingdino_f32.bmodel'
    det_pipeline = GroundingDINO(args)
else:
    raise NotImplementedError

sam_pipeline = SAM(image_encoder_path='weight/mobilesam_encoder_hwc.bmodel',
                    mask_decoder_path='weight/mask_decoder.bmodel',
                    prompt_embed_weight_dir='weight',
                    device_id=0)

# Description
title = f"<center><strong><font size='8'>Annotate and Segment Anything({'YOLOv8' if args.det_method=='yolov8s' else 'GroundingDINO'} + MobileSAM)<font></strong></center>"

description_e = f"""This is a demo of {"YOLOv8" if args.det_method=='yolov8s' else 'GroundingDINO'} + [Faster Segment Anything(MobileSAM) Model](https://github.com/ChaoningZhang/MobileSAM).

                   Enjoy!                
              """

description_p = """ # Instructions for annotation mode

                0. Restart by click the Restart button
                1. Select the object that you would like to annotate.
                2. Click the Annotate.

              """

default_example = ["./test_imgs/cat.jpg"]

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"


def seg_and_anno(src_img, anno_class, refine=True, output_json=True):
    if args.det_method == 'groundingdino':
        # decode
        det_pipeline.decode(Image.fromarray(src_img))
        # preprocess
        H, W = src_img.shape[:2]
        t0 = time.time()
        data = det_pipeline.preprocess(anno_class)
        output = det_pipeline(data)
        boxes_filt, pred_phrases = det_pipeline.postprocess(anno_class, output, W, H)
        t1 = time.time()-t0
        print('=====================GroundingDINO timecost:', t1)
        nd_image = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        sam_pipeline.set_image(nd_image)
        det = boxes_filt
        labels, boxes = [], []
        all_masks = np.zeros(src_img.shape[:2], dtype=np.int8)
        H, W = src_img.shape[:2]
        for idx in range(len(det)):
            # bbox_dict = dict()
            x1, y1, x2, y2 = det[idx]
            print(x1, y1, x2, y2)
            labels.append(anno_class)
            input_box = np.array([[x1, y1, x2, y2]])
            boxes.append(input_box[0])
            masks, iou_pred = sam_pipeline.predict(
                None, input_box, None, 
                multiple_output=not args.single_output, 
                return_logits=False)
            mask = np.array(masks[0][0])
            if refine:
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)
                )
            all_masks[mask==1] = len(labels)
        masked_img = draw_label_masks(src_img, labels, all_masks, boxes=boxes, alpha=0.5)
    elif args.det_method == 'yolov8s':
        if len(src_img.shape) != 3:
            src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2BGR)
        t0 = time.time()
        results = det_pipeline([src_img])
        t1 = time.time()-t0
        print('=====================YOLOv8s timecost:', t1)
        nd_image = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        sam_pipeline.set_image(nd_image)
        det = results[0]
        labels, boxes = [], []
        all_masks = np.zeros(src_img.shape[:2], dtype=np.int8)
        for idx in range(det.shape[0]):
            # bbox_dict = dict()
            x1, y1, x2, y2, score, category_id = det[idx]
            class_name = class_names[int(category_id+1)]
            if score < args.conf_thresh or class_name not in anno_class:
                continue
            labels.append(class_name)
            input_box = np.array([[x1, y1, x2, y2]])
            boxes.append(input_box[0])
            masks, iou_pred = sam_pipeline.predict(
                None, input_box, None, 
                multiple_output=not args.single_output, 
                return_logits=False)
            mask = np.array(masks[0][0])
            if refine:
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
                )
                mask = cv2.morphologyEx(
                    mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((8, 8), np.uint8)
                )
            all_masks[mask==1] = len(labels)
        masked_img = draw_label_masks(src_img, labels, all_masks, boxes=boxes, alpha=0.5)
    return masked_img


def seg_and_anno_video(full_video, anno_class, refine=False, output_json=True):
    cap = cv2.VideoCapture()
    if not cap.open(full_video):
        raise Exception("can not open the video")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(fps, size)
    save_video = full_video[:full_video.rfind('.')] + '_res.mp4'
    out = cv2.VideoWriter(save_video, fourcc, fps, size)
    frame_idx = 0
    while cap.isOpened():
        import time
        start_time = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        # frame_list.append(frame)
        import time; t0 = time.time()
        results = det_pipeline([frame])
        t1 = time.time()-t0
        print('=====================YOLOv8s timecost:', t1)
        nd_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t2 = time.time()
        sam_pipeline.set_image(nd_image)
        det = results[0]
        if det.shape[0] == 0:
            out.write(frame)
            print('frame {} writed'.format(frame_idx),end='\r')
            frame_idx += 1
            continue
        masks = None
        for idx in range(det.shape[0]):
            # bbox_dict = dict()
            x1, y1, x2, y2, score, category_id = det[idx]
            category_id = int(category_id+1)
            if score < args.conf_thresh or class_names[category_id] != anno_class:
                continue
            input_box = np.array([[x1, y1, x2, y2]])
            masks, iou_pred = sam_pipeline.predict(
                None, input_box, None, multiple_output=not args.single_output, return_logits=True)
            # if idx == 0:
            print('=====================SAM timecost:', time.time()-t2)
            break
        if masks is not None:
            masked_frame = draw_mask(frame, np.array(masks[0][0]), alpha=0.5)
            # masked_frame = cv2.cvtColor(masked_frame,cv2.COLOR_RGB2BGR)
            out.write(masked_frame)
        else:
            out.write(frame)
        print('frame {} writed'.format(frame_idx),end='\r')
        frame_idx += 1

    cap.release()
    out.release()
    return save_video


with gr.Blocks(css=css, title="Annotate Anything (YOLO + MobileSAM)") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)
    
    with gr.Tab('Image Annotation mode'):
        with gr.Row():
            with gr.Column():
                img_anno_inputs = [
                    gr.Image(label="Select Image", value=default_example[0], type='numpy'),
                    gr.Dropdown(label="Class", choices=class_names, multiselect=True) \
                        if args.det_method == 'yolov8s' else gr.Textbox(label="Target Description")
                ]
            with gr.Column():
                annotated_img = gr.Image(label="Annotation", interactive=False, type="numpy")

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        segment_btn_p = gr.Button(
                            "Annotate", variant="primary"
                        )
                        clear_btn_p = gr.Button("Restart", variant="secondary")


            with gr.Column():
                # Description
                gr.Markdown(description_p)
    
        segment_btn_p.click(
            seg_and_anno, inputs=img_anno_inputs, outputs=[annotated_img]#, json_str]
        )
        def clear():
            return [None, None, None]

        clear_btn_p.click(clear, outputs=[*img_anno_inputs, annotated_img])
    
    if args.det_method == 'yolov8s':
        with gr.Tab('Video Annotation mode'):
            with gr.Row():
                with gr.Column():
                    video_anno_inputs = [
                        gr.Video(label="Select Video", format='mp4'),
                        gr.Dropdown(label="Class", choices=class_names, multiselect=False) \
                            if args.det_method == 'yolov8s' else gr.Textbox(label="Target Description")
                    ]
                with gr.Column():
                    annotated_video = gr.File(label="Annotation", interactive=False)

            # Submit & Clear
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            segment_btn_p = gr.Button(
                                "Annotate", variant="primary"
                            )
                            clear_btn_p = gr.Button("Restart", variant="secondary")


                with gr.Column():
                    # Description
                    gr.Markdown(description_p)
        
            segment_btn_p.click(
                seg_and_anno_video, inputs=video_anno_inputs, outputs=[annotated_video]#, json_str]
            )
        
            def clear():
                return [None, None, None]

            clear_btn_p.click(clear, outputs=[*video_anno_inputs, annotated_video])

demo.queue()
demo.launch(ssl_verify=False, server_name="0.0.0.0")
