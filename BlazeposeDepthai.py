import numpy as np
import cv2
import struct
import socket
import time, sys

import tflite_runtime.interpreter as tflite

from pathlib import Path
from FPS import FPS, now
from math import sin, cos
import mediapipe_utils as mpu

POSE_DETECTION_TFLITE = "pose_detection.tflite"
LANDMARK_MODEL_FULL_TFLITE = "pose_landmark_full.tflite"
LANDMARK_MODEL_LITE_TFLITE = "pose_landmark_lite.tflite"
LANDMARK_MODEL_HEAVY_TFLITE = "pose_landmark_heavy.tflite"

DEFAULT_SOCKET_PATH = "/tmp/oakd_socket"

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Utility if needed (matching your original code style). 
    Typically used to convert NHWC -> CHW float arrays.
    """
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()

class BlazeposeDepthai:
    """
    Blazepose body pose detector for host-side TFLite inference.
    Frames come from a UNIX socket. No DepthAI pipeline is created.
    """

    def __init__(self, 
                 input_src="socket",
                 pd_model=None, 
                 pd_score_thresh=0.5,
                 lm_model=None,
                 lm_score_thresh=0.7,
                 xyz=False,
                 crop=False,
                 smoothing=True,
                 internal_fps=None,
                 resolution="full",
                 internal_frame_height=1080,
                 stats=False,
                 trace=False,
                 force_detection=False):

        # Store arguments
        self.input_src = input_src
        self.pd_model = pd_model if pd_model else POSE_DETECTION_TFLITE
        self.pd_score_thresh = pd_score_thresh

        if lm_model is None or lm_model == "full":
            self.lm_model = LANDMARK_MODEL_FULL_TFLITE
        elif lm_model == "lite":
            self.lm_model = LANDMARK_MODEL_LITE_TFLITE
        elif lm_model == "heavy":
            self.lm_model = LANDMARK_MODEL_HEAVY_TFLITE
        else:
            self.lm_model = lm_model

        self.lm_score_thresh = lm_score_thresh
        self.xyz = False  # Not used in host-only mode
        self.crop = crop
        self.smoothing = smoothing
        self.internal_fps = internal_fps
        self.resolution = resolution
        self.internal_frame_height = internal_frame_height
        self.stats = stats
        self.trace = trace
        self.force_detection = force_detection

        print(f"[HOST] Pose detection TFLite model: {self.pd_model}")
        print(f"[HOST] Landmark TFLite model:       {self.lm_model}")
        print("[HOST] Loading TFLite interpreters...")
        self.pd_interpreter = tflite.Interpreter(model_path=self.pd_model)
        self.pd_interpreter.allocate_tensors()
        self.lm_interpreter = tflite.Interpreter(model_path=self.lm_model)
        self.lm_interpreter.allocate_tensors()

        # PD (pose detection) I/O details
        self.pd_input_details = self.pd_interpreter.get_input_details()
        self.pd_output_details = self.pd_interpreter.get_output_details()
        self.pd_input_length = self.pd_input_details[0]['shape'][1]  # Typically 224

        # LM (landmark) I/O details
        self.lm_input_details = self.lm_interpreter.get_input_details()
        self.lm_output_details = self.lm_interpreter.get_output_details()
        self.lm_input_length = self.lm_input_details[0]['shape'][1]  # Typically 256

        # Create anchors for PD
        self.anchors = mpu.generate_blazepose_anchors()
        self.nb_anchors = self.anchors.shape[0]
        print(f"[HOST] {self.nb_anchors} anchors created for pose detection.")

        # Smoothing filters
        self.nb_kps = 33
        if self.smoothing:
            freq = 30 if not self.internal_fps else self.internal_fps
            self.filter_landmarks = mpu.LandmarksSmoothingFilter(
                frequency=freq, min_cutoff=0.05, beta=80, derivate_cutoff=1
            )
            self.filter_landmarks_aux = mpu.LandmarksSmoothingFilter(
                frequency=freq, min_cutoff=0.01, beta=10, derivate_cutoff=1
            )
            self.filter_landmarks_world = mpu.LandmarksSmoothingFilter(
                frequency=freq, min_cutoff=0.1, beta=40, derivate_cutoff=1,
                disable_value_scaling=True
            )

        self.rect_transf_scale = 1.25
        self.use_previous_landmarks = False
        self.body_from_landmarks = None

        # Stats
        self.fps = FPS()
        self.nb_frames = 0
        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.glob_pd_time = 0
        self.glob_lm_time = 0

        # Connect to the server socket
        self.socket_path = DEFAULT_SOCKET_PATH
        print(f"[HOST] Connecting to server socket: {self.socket_path}")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.socket_path)
        print("[HOST] Connected to OAK-D server socket.")

        self.read_buffer = b""

    def __del__(self):
        self.exit()

    def exit(self):
        if getattr(self, "sock", None):
            try:
                self.sock.close()
            except:
                pass
            self.sock = None
        if self.stats:
            print(f"[STATS] FPS : {self.fps.get_global():.1f} (#frames = {self.fps.nbf})")
            print(f"[STATS] # PD inferences : {self.nb_pd_inferences}")
            print(f"[STATS] # LM inferences : {self.nb_lm_inferences}")
            if self.nb_pd_inferences:
                print(f"[STATS] PD avg time : {self.glob_pd_time/self.nb_pd_inferences*1000:.2f} ms")
            if self.nb_lm_inferences:
                print(f"[STATS] LM avg time : {self.glob_lm_time/self.nb_lm_inferences*1000:.2f} ms")

    def read_frame_from_socket(self):
        """
        Reads one color frame + one depth frame from the socket.
        Returns (colorFrame, depthFrame) or (None, None) if no data.
        """
        # We expect 8 bytes: 2 x 4-byte lengths
        header_size = 8
        while len(self.read_buffer) < header_size:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None, None
            self.read_buffer += chunk

        color_len, depth_len = struct.unpack("!II", self.read_buffer[:header_size])
        self.read_buffer = self.read_buffer[header_size:]

        # Read color data
        while len(self.read_buffer) < color_len:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None, None
            self.read_buffer += chunk
        color_data = self.read_buffer[:color_len]
        self.read_buffer = self.read_buffer[color_len:]

        # Read depth data
        while len(self.read_buffer) < depth_len:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None, None
            self.read_buffer += chunk
        depth_data = self.read_buffer[:depth_len]
        self.read_buffer = self.read_buffer[depth_len:]

        colorFrame = cv2.imdecode(np.frombuffer(color_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        depthFrame = cv2.imdecode(np.frombuffer(depth_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        return colorFrame, depthFrame

    # ------------------------------------------------------
    # Pose Detection Inference with Flattening
    # ------------------------------------------------------
    def pd_infer(self, resized224):
        """
        Run TFLite pose detection. 
        *Key change:* we ensure 'scores' is flattened => shape (2254,).
        """
        input_data = resized224.astype(np.float32)

        input_data = np.expand_dims(input_data, axis=0)  # shape (1,224,224,3) if NHWC
        self.pd_interpreter.set_tensor(self.pd_input_details[0]['index'], input_data)

        t0 = time.time()
        self.pd_interpreter.invoke()
        dt = time.time() - t0
        self.glob_pd_time += dt
        self.nb_pd_inferences += 1

        out0 = self.pd_interpreter.get_tensor(self.pd_output_details[0]['index'])  # e.g. scores
        out1 = self.pd_interpreter.get_tensor(self.pd_output_details[1]['index'])  # e.g. bboxes

        # Squeeze + flatten scores => shape (2254,)
        scores = np.squeeze(out0).flatten()
        # Squeeze bboxes => shape (2254,12)
        bboxes = np.squeeze(out1)  

        return scores, bboxes

    # ------------------------------------------------------
    # Landmark Inference
    # ------------------------------------------------------
    def lm_infer(self, region256):
        """
        Run TFLite landmark model. region256 shape (256,256,3).
        Return raw arrays (lm_raw, lm_score, lm_world).
        """
        input_data = region256.astype(np.float32)
        # If [0..1] is correct, nothing else needed.

        input_data = np.expand_dims(input_data, axis=0)  # shape (1,256,256,3)
        self.lm_interpreter.set_tensor(self.lm_input_details[0]['index'], input_data)

        t0 = time.time()
        self.lm_interpreter.invoke()
        dt = time.time() - t0
        self.glob_lm_time += dt
        self.nb_lm_inferences += 1

        out0 = self.lm_interpreter.get_tensor(self.lm_output_details[0]['index'])
        out1 = self.lm_interpreter.get_tensor(self.lm_output_details[1]['index'])
        out2 = self.lm_interpreter.get_tensor(self.lm_output_details[2]['index'])

        lm_raw   = np.squeeze(out0)
        lm_score = float(np.squeeze(out1))
        lm_world = np.squeeze(out2)
        return lm_raw, lm_score, lm_world

    # ------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------
    def pd_postprocess(self, scores, bboxes, frame_size):
        bodies = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=True)
        if bodies:
            body = bodies[0]
            mpu.detections_to_rect(body)
            mpu.rect_transformation(body, frame_size, frame_size, self.rect_transf_scale)
            return body
        return None

    def lm_postprocess(self, body, lm_raw, lm_score, lm_world_raw, frame_size, square_frame):
        """
        Similar to your original logic: transform raw landmark predictions 
        back to image coords, apply smoothing, etc.
        """
        body.lm_score = lm_score
        print(f"Landmark model score={lm_score:.3f}, threshold={self.lm_score_thresh}")
        if lm_score < self.lm_score_thresh:
            return None

        # Expect shape 195 => (39,5), for 33 visible + 2 ROI + possibly extra
        lm_2d = lm_raw.reshape(-1, 5)
        lm_2d[:,:3] /= self.lm_input_length

        # Sigmoid on presence & visibility
        visibility = 1.0 / (1.0 + np.exp(-lm_2d[:,3]))
        presence   = 1.0 / (1.0 + np.exp(-lm_2d[:,4]))
        body.visibility = visibility
        body.presence   = presence
        body.norm_landmarks = lm_2d[:,:3]

        # Warp back to original square frame
        src = np.array([(0,0), (1,0), (1,1)], dtype=np.float32)
        dst = np.array(body.rect_points[1:], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        lm_xy = np.expand_dims(body.norm_landmarks[:self.nb_kps+2, :2], axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat))
        lm_z  = body.norm_landmarks[:self.nb_kps+2, 2:3] * body.rect_w_a / 4
        lm_xyz = np.hstack((lm_xy, lm_z))

        if self.smoothing:
            timestamp = now()
            object_scale = body.rect_w_a
            lm_xyz[:self.nb_kps]   = self.filter_landmarks.apply(lm_xyz[:self.nb_kps], timestamp, object_scale)
            lm_xyz[self.nb_kps:]   = self.filter_landmarks_aux.apply(lm_xyz[self.nb_kps:], timestamp, object_scale)

        body.landmarks = lm_xyz.astype(np.int32)

        # World landmarks
        wlm = lm_world_raw.reshape(-1,3)[:self.nb_kps]
        sin_rot = sin(body.rotation)
        cos_rot = cos(body.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        wlm[:,:2] = np.dot(wlm[:,:2], rot_m)
        if self.smoothing:
            timestamp = now()
            wlm = self.filter_landmarks_world.apply(wlm, timestamp)
        body.landmarks_world = wlm

        # Save the last 2 kps as ROI for next frame
        self.body_from_landmarks = mpu.Body(
            pd_kps=body.landmarks[self.nb_kps:self.nb_kps+2,:2] / frame_size
        )
        return body

    # ------------------------------------------------------
    # Main loop: next_frame
    # ------------------------------------------------------
    def next_frame(self):
        self.fps.update()
        self.nb_frames += 1

        # 1) Receive frames from the socket
        colorFrame, depthFrame = self.read_frame_from_socket()
        if colorFrame is None:
            # No more frames or socket closed
            return None, None

        h, w, _ = colorFrame.shape
        if self.crop:
            frame_size = min(h, w)
            if w > h:
                pad = (w - h)//2
                square_frame = colorFrame[:, pad:pad+frame_size]
            else:
                pad = (h - w)//2
                square_frame = colorFrame[pad:pad+frame_size, :]
        else:
            if w > h:
                pad_vertical = (w - h)//2
                square_frame = cv2.copyMakeBorder(
                    colorFrame,
                    pad_vertical, pad_vertical, 0, 0,
                    cv2.BORDER_CONSTANT, value=(0,0,0)
                )
                frame_size = w
            else:
                pad_horizontal = (h - w)//2
                square_frame = cv2.copyMakeBorder(
                    colorFrame,
                    0, 0, pad_horizontal, pad_horizontal,
                    cv2.BORDER_CONSTANT, value=(0,0,0)
                )
                frame_size = h

        # 2) Pose detection if forced or no previous landmarks
        if self.force_detection or not self.use_previous_landmarks:
            pd_input = cv2.resize(square_frame, (self.pd_input_length, self.pd_input_length))
            pd_input = pd_input.astype(np.float32)/255.0
            scores, bboxes = self.pd_infer(pd_input)
            body = self.pd_postprocess(scores, bboxes, frame_size)
        else:
            # Reuse bounding box from the previous frame
            body = self.body_from_landmarks
            if body:
                mpu.detections_to_rect(body)
                mpu.rect_transformation(body, frame_size, frame_size, self.rect_transf_scale)

        if body is None:
            print("Body is none!")
            self.use_previous_landmarks = False
            return colorFrame, None

        # 3) Landmark inference
        region256 = mpu.warp_rect_img(body.rect_points, square_frame,
                                      self.lm_input_length, self.lm_input_length)
        region256 = region256.astype(np.float32)/255.0
        lm_raw, lm_score, lm_world = self.lm_infer(region256)
        final_body = self.lm_postprocess(body, lm_raw, lm_score, lm_world, frame_size, square_frame)
        if final_body is None:
            print("Final body is none!")
            self.use_previous_landmarks = False
            return colorFrame, None

        self.use_previous_landmarks = True

        
        return colorFrame, final_body
