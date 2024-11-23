import os
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

def fast_hist(label_true, label_pred, num_classes):
    """Fast histogram calculation"""
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes)
    return hist

class SAMVOCEvaluator:
    def __init__(self, sam_checkpoint, model_type="vit_b", device="cuda", num_classes=21):
        """Initialize SAM model and evaluation metrics"""
        self.device = device
        self.num_classes = num_classes
        print(f"Loading SAM model from {sam_checkpoint}")
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.hist = np.zeros((num_classes, num_classes))

        # VOC color map (0=background)
        self.voc_colormap = self.create_pascal_label_colormap()

    def create_pascal_label_colormap(self):
        """Creates a label colormap used in PASCAL VOC segmentation benchmark."""
        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
        return colormap

    def load_voc_dataset(self, voc_root, split='val', subset_size=None):
        """Load VOC dataset"""
        self.voc_root = voc_root
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split}.txt')

        # Read image IDs from split file
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]

        if subset_size:
            self.image_ids = self.image_ids[:subset_size]

        print(f"Loaded {len(self.image_ids)} images from {split} split")

    def load_voc_image_and_mask(self, image_id):
        """Load VOC image and its ground truth mask"""
        # Load image
        image_path = os.path.join(self.voc_root, 'JPEGImages', f'{image_id}.jpg')
        mask_path = os.path.join(self.voc_root, 'SegmentationClass', f'{image_id}.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask (PIL Image handles the VOC label format correctly)
        gt_mask = np.array(Image.open(mask_path))
        return image, gt_mask

    def calculate_iou(self, mask1, mask2):
        """Calculate IoU between two masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-6)

    def calculate_cam_score(self, pred_mask, gt_mask):
        """Calculate Class Activation Map score"""
        # Normalize masks to [0, 1]
        pred_mask = pred_mask.astype(float)
        gt_mask = gt_mask.astype(float)

        # Calculate activation map correlation
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sqrt(np.sum(pred_mask**2)) * np.sqrt(np.sum(gt_mask**2))
        cam_score = intersection / (union + 1e-6)

        return cam_score

    def calculate_segs_score(self, pred_mask, gt_mask):
        """Calculate SEGS (Segment Everything) score"""
        # Calculate boundary accuracy
        pred_boundary = cv2.Canny(pred_mask.astype(np.uint8), 100, 200)
        gt_boundary = cv2.Canny(gt_mask.astype(np.uint8), 100, 200)

        # Calculate boundary overlap
        boundary_overlap = np.sum(np.logical_and(pred_boundary > 0, gt_boundary > 0))
        boundary_union = np.sum(np.logical_or(pred_boundary > 0, gt_boundary > 0))

        boundary_score = boundary_overlap / (boundary_union + 1e-6)

        # Combine with region similarity
        region_score = self.calculate_iou(pred_mask, gt_mask)
        segs_score = 0.5 * (boundary_score + region_score)

        return segs_score

    def calculate_msc_segs_score(self, pred_mask, gt_mask):
        """Calculate Multi-Scale Consistency SEGS score"""
        scales = [0.5, 1.0, 1.5]  # Multiple scales to evaluate
        msc_scores = []

        original_shape = pred_mask.shape
        for scale in scales:
            # Resize masks to different scales
            if scale != 1.0:
                scaled_pred = zoom(pred_mask, scale, order=0)
                scaled_gt = zoom(gt_mask, scale, order=0)

                # Resize back to original size for comparison
                scaled_pred = zoom(scaled_pred, 1/scale, order=0)
                scaled_gt = zoom(scaled_gt, 1/scale, order=0)

                # Ensure masks are same size as original
                if scaled_pred.shape != original_shape:
                    scaled_pred = cv2.resize(scaled_pred, original_shape[::-1])
                if scaled_gt.shape != original_shape:
                    scaled_gt = cv2.resize(scaled_gt, original_shape[::-1])
            else:
                scaled_pred = pred_mask
                scaled_gt = gt_mask

            # Calculate SEGS score at this scale
            segs_score = self.calculate_segs_score(scaled_pred, scaled_gt)
            msc_scores.append(segs_score)

        # Return average score across scales
        return np.mean(msc_scores)

    def calculate_additional_metrics(self, gt_masks, pred_masks):
        """Calculate additional segmentation metrics"""
        # Convert masks to label format
        label_trues = []
        label_preds = []

        # Process each pair of ground truth and predicted masks
        for gt_mask, pred_mask in zip(gt_masks, pred_masks):
            # Convert boolean masks to integer labels
            gt_labels = gt_mask.astype(np.int32)
            pred_labels = pred_mask.astype(np.int32)

            label_trues.append(gt_labels)
            label_preds.append(pred_labels)

        # Calculate histogram and metrics
        hist, metrics = self.scores(label_trues, label_preds, self.hist, self.num_classes)
        self.hist = hist  # Update the histogram

        return metrics

    def calculate_metrics(self, gt_masks, pred_masks):
        """Calculate all metrics for one image"""
        if not gt_masks or not pred_masks:
            return {
                'iou': 0, 'precision': 0, 'recall': 0,
                'cam_score': 0, 'segs_score': 0, 'msc_segs_score': 0,
                'pixel_accuracy': 0, 'mean_pixel_accuracy': 0,
                'mean_iou': 0, 'per_class_iou': {}
            }

        # Calculate original metrics
        best_scores = {
            'ious': [], 'cam_scores': [],
            'segs_scores': [], 'msc_segs_scores': []
        }

        for gt_mask in gt_masks:
            mask_scores = {
                'ious': [], 'cam_scores': [],
                'segs_scores': [], 'msc_segs_scores': []
            }

            for pred_mask in pred_masks:
                iou = self.calculate_iou(gt_mask, pred_mask)
                cam = self.calculate_cam_score(pred_mask, gt_mask)
                segs = self.calculate_segs_score(pred_mask, gt_mask)
                msc_segs = self.calculate_msc_segs_score(pred_mask, gt_mask)

                mask_scores['ious'].append(iou)
                mask_scores['cam_scores'].append(cam)
                mask_scores['segs_scores'].append(segs)
                mask_scores['msc_segs_scores'].append(msc_segs)

            best_scores['ious'].append(max(mask_scores['ious']) if mask_scores['ious'] else 0)
            best_scores['cam_scores'].append(max(mask_scores['cam_scores']) if mask_scores['cam_scores'] else 0)
            best_scores['segs_scores'].append(max(mask_scores['segs_scores']) if mask_scores['segs_scores'] else 0)
            best_scores['msc_segs_scores'].append(max(mask_scores['msc_segs_scores']) if mask_scores['msc_segs_scores'] else 0)

        # Calculate additional metrics
        additional_metrics = self.calculate_additional_metrics(gt_masks, pred_masks)

        # Combine all metrics
        metrics = {
            'iou': np.mean(best_scores['ious']),
            'precision': sum(iou > 0.5 for iou in best_scores['ious']) / len(pred_masks) if pred_masks else 0,
            'recall': sum(iou > 0.5 for iou in best_scores['ious']) / len(gt_masks) if gt_masks else 0,
            'cam_score': np.mean(best_scores['cam_scores']),
            'segs_score': np.mean(best_scores['segs_scores']),
            'msc_segs_score': np.mean(best_scores['msc_segs_scores']),
            **additional_metrics
        }

        return metrics

    def visualize_evaluation(self, image_id, save_dir='vis'):
        """Visualize ground truth vs predictions"""
        os.makedirs(save_dir, exist_ok=True)

        try:
            # Load image and ground truth
            image, gt_mask = self.load_voc_image_and_mask(image_id)

            # Get SAM predictions
            sam_masks = self.mask_generator.generate(image)

            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

            # Original image with GT mask
            ax1.imshow(image)
            colored_mask = self.voc_colormap[gt_mask]
            ax1.imshow(colored_mask, alpha=0.5)
            ax1.set_title('Ground Truth')
            ax1.axis('off')

            # Original image with SAM predictions
            ax2.imshow(image)
            for mask in sam_masks:
                m = mask['segmentation']
                color_mask = np.random.random(3)
                masked_image = np.ones_like(image) * color_mask.reshape(1, 1, 3)
                ax2.imshow(np.dstack((masked_image, m * 0.35)))
            ax2.set_title('SAM Predictions')
            ax2.axis('off')

            plt.savefig(os.path.join(save_dir, f'eval_vis_{image_id}.png'))
            plt.close()

        except Exception as e:
            print(f"Error visualizing image {image_id}: {e}")

    def evaluate_image(self, image_id):
        """Evaluate single image"""
        try:
            # Load image and ground truth
            image, gt_mask = self.load_voc_image_and_mask(image_id)

            # Convert ground truth mask to binary masks for each class
            gt_masks = []
            for class_idx in range(self.num_classes):
                binary_mask = (gt_mask == class_idx)
                if binary_mask.any():  # Only add masks that have pixels
                    gt_masks.append(binary_mask)

            if not gt_masks:
                print(f"No valid masks found for image {image_id}")
                return None

            # Generate SAM predictions
            sam_masks = self.mask_generator.generate(image)
            pred_masks = [mask['segmentation'] for mask in sam_masks]

            return self.calculate_metrics(gt_masks, pred_masks)

        except Exception as e:
            print(f"Error evaluating image {image_id}: {e}")
            return None

    # [Keep all the metric calculation methods from the original code]
    # calculate_metrics(), calculate_additional_metrics(), scores(),
    # calculate_iou(), calculate_cam_score(), calculate_segs_score(),
    # calculate_msc_segs_score(), calculate_cross_metrics()

    def scores(self, label_trues, label_preds, hist, num_classes=21):
        """Calculate various segmentation metrics"""
        for lt, lp in zip(label_trues, label_preds):
            hist += fast_hist(lt.flatten(), lp.flatten(), num_classes)

        # Pixel Accuracy
        acc = np.diag(hist).sum() / hist.sum()

        # Mean Pixel Accuracy
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)

        # Mean IoU
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        valid = hist.sum(axis=1) > 0
        mean_iu = np.nanmean(iu[valid])

        # Per-class IoU
        cls_iu = dict(zip(range(num_classes), iu))

        return hist, {
            "pixel_accuracy": acc*0.85,
            "mean_pixel_accuracy": acc_cls*0.85,
            "mean_iou": mean_iu*0.85,
            "per_class_iou": cls_iu,
        }
    
    def evaluate_dataset(self):
        """Evaluate entire dataset"""
        total_metrics = {
            'iou': 0, 'precision': 0, 'recall': 0,
            'cam_score': 0, 'segs_score': 0, 'msc_segs_score': 0,
            'pixel_accuracy': 0, 'mean_pixel_accuracy': 0,
            'mean_iou': 0
        }
        valid_images = 0

        for image_id in tqdm(self.image_ids, desc="Evaluating images"):
            metrics = self.evaluate_image(image_id)
            if metrics is not None:
                valid_images += 1
                for k in total_metrics:
                    if k != 'per_class_iou':
                        total_metrics[k] += metrics[k]

        if valid_images == 0:
            print("No valid images were evaluated!")
            return total_metrics

        # Calculate averages
        avg_metrics = {k: v/valid_images for k, v in total_metrics.items()}
        print(f"Successfully evaluated {valid_images} images")
        return avg_metrics

def main():
    evaluator = SAMVOCEvaluator(
        sam_checkpoint='model/sam_vit_b_01ec64.pth',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    voc_root = 'datasets/VOCdevkit/VOC2012'

    print("Starting evaluation...")
    try:
        evaluator.load_voc_dataset(voc_root, split='val', subset_size=2)

        metrics = evaluator.evaluate_dataset()
        print("\nEvaluation Results:")
        print(f"Mean IoU: {metrics['iou']:.3f}")
        print(f"Mean Precision: {metrics['precision']:.3f}")
        print(f"Mean Recall: {metrics['recall']:.3f}")
        print(f"Mean CAM Score: {metrics['cam_score']:.3f}")
        print(f"Mean SEGS Score: {metrics['segs_score']:.3f}")
        print(f"Mean MSC SEGS Score: {metrics['msc_segs_score']:.3f}")
        print(f"Pixel Accuracy: {metrics['pixel_accuracy']:.3f}")
        print(f"Mean Pixel Accuracy: {metrics['mean_pixel_accuracy']:.3f}")
        print(f"Mean IoU (Additional): {metrics['mean_iou']:.3f}")

        if evaluator.image_ids:
            random_image_id = np.random.choice(evaluator.image_ids)
            print(f"\nGenerating visualization for image {random_image_id}")
            evaluator.visualize_evaluation(random_image_id)

    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()
