import torch
import numpy as np
import os
import cv2
import sophon.sail as sail
from typing import List, Optional, Tuple, Type
from torch.nn import functional as F
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from .sam_utils.transforms import ResizeLongestSide, Padding2Normal
import PIL.Image as Image


class EngineOV:

    def __init__(self, model_path="", output_names="", device_id=0):
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = ", device_id)
        self.model_path = model_path
        self.device_id = device_id
        try:
            self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        except Exception as e:
            print("load model error; please check model path and device status;")
            print(">>>> model_path: ", model_path)
            print(">>>> device_id: ", device_id)
            print(">>>> sail.Engine error: ", e)
            raise e
        sail.set_print_flag(True)
        self.graph_name = self.model.get_graph_names()[0]
        self.input_name = self.model.get_input_names(self.graph_name)
        self.output_name = self.model.get_output_names(self.graph_name)

    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path, self.device_id)

    def __call__(self, args):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        args = {}
        for i in range(len(values)):
            args[self.input_name[i]] = values[i]
        output = self.model.process(self.graph_name, args)
        res = []

        for name in self.output_name:
            res.append(output[name])
        return res


class SAM:
    def __init__(self,
                 image_encoder_path: str,
                 mask_decoder_path: str,
                 #  single_mask_decoder_path:str,
                 prompt_embed_weight_dir: str,
                 img_size: Tuple[int, int] = (1024, 1024),
                 use_point=True,
                 use_box=False,
                 use_mask=False,
                 device_id=0) -> None:
        self.img_size = img_size
        self.prompt_embed_weight_dir = prompt_embed_weight_dir
        self.image_encoder = EngineOV(image_encoder_path, device_id=device_id)
        self.multi_mask_decoder = EngineOV(
            mask_decoder_path, device_id=device_id)
        # num_point_embeddings = 4  # pos/neg point + 2 box corners
        self.prompt_encoder_weight = {'positional_encoding_gaussian_matrix': np.load(os.path.join(self.prompt_embed_weight_dir, "positional_encoding_gaussian_matrix.npy")),
                                      'not_a_point_embed_weight': np.load(os.path.join(self.prompt_embed_weight_dir, "not_a_point_embed_weight.npz"))['arr_0'],
                                      'point_embedding_weights': np.load(os.path.join(self.prompt_embed_weight_dir, "point_embeddings.npz"))['arr_0'],
                                      'no_mask_embed_weight': np.load(os.path.join(self.prompt_embed_weight_dir, "no_mask_embed_weight.npz"))['arr_0']
                                      }
        self.transform1 = ResizeLongestSide(img_size[0])
        self.transform2 = Padding2Normal(img_size[0])
        self.mask_threshold = 0.0
        self.reset_image()

    def encoding_prompt(self, points, boxes, mask_inputs, prompt_embed_dim=256, batch_size=1, pad_to=5):
        point_prompts, box_prompts, mask_prompts = self.prepare_prompts(
            points, boxes, mask_inputs)
        image_size = self.img_size[0]
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        bs = batch_size # points[0].shape[0]  # self._get_batch_size
        sparse_embeddings = torch.empty((bs, 0, prompt_embed_dim))
        import time; st_time = time.time()
        if point_prompts is not None:
            point_embeddings = self.get_point_prompt_encoding(*(point_prompts[:2]))
            print("point encoding timecost: ", time.time()-st_time)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if box_prompts is not None:
            # box_prompts: A Bx4 array given a box prompt to the model, in XYXY format
            box_embeddings = self.get_box_prompt_encoding(box_prompts)
            print("box encoding timecost: ", time.time()-st_time)
            sparse_embeddings = torch.cat(
                [sparse_embeddings, box_embeddings], dim=1)
        dense_pe = self.get_dense_pe(
            (image_embedding_size, image_embedding_size)).unsqueeze(0)
        if mask_inputs is not None:
            raise NotImplementedError
        else:
            dense_embeddings = np.broadcast_to(np.reshape(self.prompt_encoder_weight['no_mask_embed_weight'], (1, -1, 1, 1)),
                                               (bs, dense_pe.shape[1], image_embedding_size, image_embedding_size)).copy()
        if pad_to is not None:
            padding_embeddings = np.zeros((bs, pad_to-sparse_embeddings.shape[1], prompt_embed_dim))
            padding_embeddings[:,:] = self.prompt_encoder_weight['not_a_point_embed_weight']
            sparse_embeddings = np.concatenate([sparse_embeddings, padding_embeddings], axis=1)
        
        return dense_pe, sparse_embeddings, dense_embeddings

    def get_box_prompt_encoding(self, boxes: np.array):
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        coords = torch.FloatTensor(coords)
        coords[:, :, 0] = coords[:, :, 0] / self.img_size[1]  # / ori_img_size[1]
        coords[:, :, 1] = coords[:, :, 1] / self.img_size[0]  # / ori_img_size[0]
        corner_embedding = self.pe_encoding(coords)
        corner_embedding[:, 0, :] += self.prompt_encoder_weight['point_embedding_weights'][2]
        corner_embedding[:, 1, :] += self.prompt_encoder_weight['point_embedding_weights'][3]
        return corner_embedding

    def get_point_prompt_encoding(self, points, labels):
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        # if boxes is not None: xxx
        """Positionally encode points that are not normalized to [0,1]."""
        coords = torch.FloatTensor(points)  # points
        labels = torch.IntTensor(labels)
        
        # 坐标处理要和prepare_img对照
        coords[:, :, 0] = coords[:, :, 0] / self.img_size[1]  # / ori_img_size[1]
        coords[:, :, 1] = coords[:, :, 1] / self.img_size[0]  # / ori_img_size[0]
        point_embeddings = self.pe_encoding(coords)

        point_embeddings[labels == 0] += self.prompt_encoder_weight['point_embedding_weights'][0]
        point_embeddings[labels == 1] += self.prompt_encoder_weight['point_embedding_weights'][1]

        return point_embeddings

    def prepare_prompts(self, points, boxes, mask_input):
        processed_points, processed_boxes = None, None
        # Transform input prompts
        if points is not None:
            point_coords, point_labels = points
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform1.apply_coords(points[0], self.original_size)
            processed_points = (point_coords[np.newaxis, :], point_labels[np.newaxis, :])
        if boxes is not None:
            boxes = self.transform1.apply_boxes(boxes, self.original_size)
            processed_boxes = boxes[None, :]
        if mask_input is not None:
            raise NotImplementedError
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float)
            mask_input_torch = mask_input_torch[None, :, :, :]
        return processed_points, processed_boxes, None,

    def set_image(self, ori_image: np.ndarray) -> np.ndarray:
        self.reset_image()
        self.original_size = ori_image.shape[:2]
        transformed_image = self.transform1.apply_image(ori_image)
        self.input_size = tuple(transformed_image.shape[:2])
        input_image = self.transform2.apply_image_bmodel(transformed_image)
        import time; st_time = time.time()
        self.features = self.image_encoder([input_image])[0]
        print("====================image encoding timecost:", time.time()-st_time)
        self.is_image_set = True

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    def predict(self, points, boxes, mask_inputs, multiple_output=False, return_logits=False):
        """
        Predict masks for the given input prompts, using the currently set image.
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask prediction.")
        import time;st_time = time.time()
        dense_pe, sparse_embeddings, dense_embeddings = self.encoding_prompt(points, boxes, mask_inputs)
        print("=================prompt encoding timecost: ", time.time()-st_time)
        st_time = time.time()
        masks, iou_pred = self.multi_mask_decoder([self.features, dense_pe, sparse_embeddings, dense_embeddings])
        print("=================mask decoding timecost: ", time.time()-st_time)
        if multiple_output:
            masks = self.postprocess_masks(masks, self.input_size, self.original_size)
        else:
            masks = masks[0][np.argmax(iou_pred)].reshape(1, 1, masks.shape[2], masks.shape[3])
            iou_pred = np.max(iou_pred)
            masks = self.postprocess_masks(masks, self.input_size, self.original_size)[0]
        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_pred  # , annotations

    def postprocess_masks(
        self,
        masks,
        input_size,
        original_size
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            torch.from_numpy(masks),
            self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size,
                              mode="bilinear", align_corners=False)
        return masks

    def pe_encoding(self, coords):
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.prompt_encoder_weight['positional_encoding_gaussian_matrix']
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        # B x N x C
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def get_dense_pe(self, img_emb_size) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = img_emb_size
        grid = torch.ones((h, w), dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self.pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


_palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
_palette = [0,0,0]+_palette


# cv2.rectangle()
# 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
# cv2.rectangle(img, (x,y), (x+w,y+h), (B,G,R), Thickness)
 
# cv2.putText()
# 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
# cv2.putText(img, text, (x,y), Font, Size, (B,G,R), Thickness)
def draw_label_masks(img, labels, all_masks, boxes=None, alpha=0.5):
    img_mask = np.zeros_like(img)
    img_mask = img
    for i, label in enumerate(labels):
        id = i + 1
        # Overlay color on  binary mask
        if id <= 255:
            color = _palette[id*3:id*3+3]
        else:
            color = [0,0,0]
        foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
        binary_mask = (all_masks == id)

        # Compose image
        img_mask[binary_mask] = foreground[binary_mask]

        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        img_mask[countours, :] = 0
        if i == 0:
            img_mask = cv2.putText(img_mask, label, (int((boxes[i][0]+boxes[i][2])/2), int((boxes[i][1]+boxes[i][3])/2)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2)
    return img_mask.astype(img.dtype)


def draw_mask(img, mask, alpha=0.5, id_countour=False):
    def colorize_mask(pred_mask):
        save_mask = Image.fromarray(pred_mask.astype(np.uint8))
        save_mask = save_mask.convert(mode='P')
        save_mask.putpalette(_palette)
        save_mask = save_mask.convert(mode='RGB')
        return np.array(save_mask)
    
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask) #sorted
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask>0)
        # import pdb;pdb.set_trace()
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(binary_mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)


def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                 facecolor=(0, 0, 0, 0), lw=2))


if __name__ == "__main__":
    torch.manual_seed(7)

    sam_pipeline = SAM(image_encoder_path='weight/mobilesam_encoder_hwc.bmodel',
                       mask_decoder_path='weight/mask_decoder.bmodel',
                       prompt_embed_weight_dir='weight',
                       device_id=0)

    model_input_size = 1024
    input_img_PIL = Image.open('test_imgs/cat.jpg')
    nd_image = np.array(input_img_PIL)
    sam_pipeline.set_image(nd_image)

    input_point = [[256, 256]]
    input_label = [1]

    masks, iou_pred = sam_pipeline.predict(
        (np.array(input_point), np.array(input_label)), None, None, multiple_output=True)
    print(iou_pred)
    # # print(input_img.shape, masks.shape, iou_pred)
    input_point = np.array(input_point)
    input_label = np.array(input_label)

    masks = masks > 0
    for i in range(masks.shape[1]):
        plt.figure(figsize=(10, 10))
        # plt.imshow(nd_image)
        show_mask(masks[0][i], plt.gca())
        show_points(input_point[0], input_label[0], plt.gca())
        plt.savefig(f'debug/point_mask_{i}.png')

    # input_point = [[800, 1000], [500,1000], [1250,1000]]
    # input_label = [1,1,1]

    # best_mask, iou = sam_pipeline.predict(
    #     (np.array(input_point), np.array(input_label)), None, None)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(nd_image)
    # show_mask(best_mask, plt.gca())
    # input_point = np.array(input_point)
    # input_label = np.array(input_label)
    # show_points(input_point[0], input_label[0], plt.gca())
    # plt.savefig(f'best_mask.png')
    
    
    input_box = np.array([[100, 150, 500, 400]])
    masks, iou_pred = sam_pipeline.predict(
        None, input_box, None, multiple_output=True)
    print(iou_pred)
    # # print(input_img.shape, masks.shape, iou_pred)
    input_box = np.array(input_box)

    masks = masks > 0
    for i in range(masks.shape[1]):
        plt.figure(figsize=(10, 10))
        plt.imshow(nd_image)
        show_mask(masks[0][i], plt.gca())
        show_box(input_box[0], plt.gca())
        plt.savefig(f'debug/box_mask_{i}.png')
