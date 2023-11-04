import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
             'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
             'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
new_kpoints = keypoints
keypoint_colors = {
    'nose': (255, 192, 203),  # Pink
    'left_eye': (255, 255, 0),  # Yellow
    'right_eye': (255, 255, 0),  # Yellow
    'left_ear': (255, 255, 0),  # Yellow
    'right_ear': (255, 255, 0),  # Yellow
    'left_shoulder': (0, 255, 0),  # Green
    'right_shoulder': (0, 255, 0),  # Green
    'left_elbow': (0, 255, 255),  # Cyan
    'right_elbow': (0, 255, 255),  # Cyan
    'left_wrist': (255, 0, 255),  # Magenta
    'right_wrist': (255, 0, 255),  # Magenta
    'left_hip': (0, 0, 255),  # Blue
    'right_hip': (0, 0, 255),  # Blue
    'left_knee': (255, 255, 255),  # White
    'right_knee': (255, 255, 255),  # White
    'left_ankle': (128, 0, 128),  # Purple
    'right_ankle': (128, 0, 128)  # Purple
}


def get_limbs_from_keypoints(keypoints):
    limbs = [
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
        [keypoints.index('right_shoulder'), keypoints.index('right_hip')],
        [keypoints.index('left_shoulder'), keypoints.index('left_hip')]
    ]
    return limbs


limbs = get_limbs_from_keypoints(keypoints)

def get_skelton_image(img):
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(img)
    output = model([img_tensor])[0]

    def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
            img_copy = np.zeros_like(img)
            for person_id in range(len(all_keypoints)):
                if confs[person_id] > conf_threshold:
                    keypoints = all_keypoints[person_id, ...]
                    scores = all_scores[person_id, ...]
                    for kp in range(len(scores)):
                        if scores[kp] > keypoint_threshold:
                            keypoint = tuple(
                                map(int, keypoints[kp, :2].detach().numpy().tolist()))
                            color = keypoint_colors[new_kpoints[kp]]
                            cv2.circle(img_copy, keypoint, 20, color, -1)
            return img_copy

    keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"], keypoint_threshold=2)

    def draw_skeleton_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
        cmap = plt.get_cmap('rainbow')
        if len(output["keypoints"]) > 0:
            colors = np.arange(
                1, 255, 255 // len(all_keypoints)).tolist()[::-1]
            for person_id in range(len(all_keypoints)):
                if confs[person_id] > conf_threshold:
                    keypoints = all_keypoints[person_id, ...]
                    for limb_id in range(len(limbs)):
                        limb_loc1 = keypoints[limbs[limb_id][0], :2].detach(
                        ).numpy().astype(np.int32)
                        limb_loc2 = keypoints[limbs[limb_id][1], :2].detach(
                        ).numpy().astype(np.int32)
                        limb_score = min(
                            all_scores[person_id, limbs[limb_id][0]], all_scores[person_id, limbs[limb_id][1]])
                        if limb_score > keypoint_threshold:
                            color = tuple(np.asarray(
                                cmap(colors[person_id])[:-1]) * 255)
                            cv2.line(keypoints_img, tuple(
                                limb_loc1), tuple(limb_loc2), color, 15)
        return keypoints_img

    skeleton_img = draw_skeleton_per_person(
        img, output["keypoints"], output["keypoints_scores"], output["scores"], keypoint_threshold=2)

    return Image.fromarray(skeleton_img,'RGB')
    