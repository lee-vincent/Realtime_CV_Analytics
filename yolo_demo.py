# --- Add this block at the very top ---
import sys
import os

# Construct the absolute path to the venv site-packages
# Use the known absolute path for simplicity in this case
# venv_site_packages = 'D:\\Code\\DNN\\dnn_env\\lib\\site-packages'
venv_site_packages = 'C:\\Program Files\\NVIDIA\\CUDNN\\v9.8\\lib\\x64'


# Check if the path exists and insert it at the beginning if it's not already first
if os.path.exists(venv_site_packages) and (not sys.path or sys.path[0] != venv_site_packages):
    # Remove it if it exists elsewhere in the path to avoid duplicates
    while venv_site_packages in sys.path:
        sys.path.remove(venv_site_packages)
    # Insert it at the beginning
    sys.path.insert(0, venv_site_packages)
    print(f"INFO: Inserted '{venv_site_packages}' at the start of sys.path")

print("--- sys.path after modification ---")
print(sys.path)
# -----------------------------------------

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA context
# Note: Second print(sys.path) removed as redundant

# --- TODO: Define your class names here (e.g., for COCO dataset) ---
# Example for COCO:
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#CLASS_NAMES = None # Replace None with your list of class names


# --- Helper Functions --

def preprocess_image(image, input_shape):
    """Preprocesses an image for YOLOv5 TensorRT input."""
    # Resize
    img = cv2.resize(image, input_shape) # target shape is (width, height) for resize
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # HWC to CHW (Height, Width, Channels to Channels, Height, Width)
    img = img.transpose((2, 0, 1))
    # Normalize to [0, 1] and make contiguous
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    return img

def postprocess_output(output, input_shape_wh, original_shape_hw, conf_thres=0.25, iou_thres=0.45):
    """Postprocesses the YOLOv5 output to get bounding boxes, scores, and class IDs."""
    predictions = np.squeeze(output) # Remove batch dimension if present (e.g., (1, 25200, 85) -> (25200, 85))

    # Filter out low-confidence predictions (objectness score)
    obj_conf = predictions[:, 4]
    predictions = predictions[obj_conf >= conf_thres]

    if len(predictions) == 0:
        return []

    # Calculate class scores
    class_scores = predictions[:, 5:] * predictions[:, 4:5] # Multiply class probs by objectness score
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Filter based on final confidence
    valid_indices = confidences >= conf_thres
    predictions = predictions[valid_indices]
    confidences = confidences[valid_indices]
    class_ids = class_ids[valid_indices]

    if len(predictions) == 0:
        return []

    # Box coordinates are (center_x, center_y, width, height) normalized to input_shape
    box_coords = predictions[:, :4]

    # Scale boxes back to original image dimensions
    orig_h, orig_w = original_shape_hw
    input_w, input_h = input_shape_wh

    # Ratio for scaling
    ratio_w = orig_w / input_w
    ratio_h = orig_h / input_h

    # Convert (center_x, center_y, width, height) to (left, top, width, height)
    center_x = box_coords[:, 0] * ratio_w
    center_y = box_coords[:, 1] * ratio_h
    width = box_coords[:, 2] * ratio_w
    height = box_coords[:, 3] * ratio_h

    left = (center_x - width / 2).astype(int)
    top = (center_y - height / 2).astype(int)
    width = width.astype(int)
    height = height.astype(int)

    # Clip boxes to image boundaries
    left = np.clip(left, 0, orig_w - 1)
    top = np.clip(top, 0, orig_h - 1)
    width = np.clip(width, 1, orig_w - left) # Ensure width >= 1
    height = np.clip(height, 1, orig_h - top) # Ensure height >= 1

    boxes_xywh = list(zip(left, top, width, height)) # List of (l, t, w, h) tuples

    # Apply Non-Maximum Suppression (NMS)
    # cv2.dnn.NMSBoxes requires boxes in (x, y, w, h) format and returns indices to keep
    indices = cv2.dnn.NMSBoxes(boxes_xywh, confidences.tolist(), conf_thres, iou_thres)

    detections = []
    # Ensure indices is not empty and is iterable
    if len(indices) > 0:
        # If indices is a flat array (common from NMSBoxes), iterate directly
        if isinstance(indices, np.ndarray):
             indices = indices.flatten()
        for i in indices:
            box = boxes_xywh[i]
            score = confidences[i]
            class_id = class_ids[i]
            detections.append((box, score, class_id))

    return detections


# --- TensorRT Engine Loading and Inference ---

class TRTInference:
    """Handles TensorRT engine loading and inference."""

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"Error loading TensorRT engine: {e}")
            raise e

        if not self.engine:
            raise RuntimeError("Failed to deserialize CUDA engine.")

        self.context = self.engine.create_execution_context()
        if not self.context:
             raise RuntimeError("Failed to create execution context.")

        # Allocate buffers based on engine bindings
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        # NOTE: self.stream is allocated but won't be explicitly used with execute_v2

    def allocate_buffers(self):
        """Allocates memory for input and output host/device buffers."""
        inputs = []
        outputs = []
        bindings = []
        # Stream is still created for memory copies, even if not used by execute_v2
        stream = cuda.Stream()

        for binding in self.engine:
            # Get shape and dtype for the current binding
            shape = self.engine.get_tensor_shape(binding)
            dtype = trt.nptype(self.engine.get_tensor_dtype(binding))

            if -1 in shape:
                 print(f"Warning: Binding '{binding}' has a dynamic shape: {shape}. "
                       f"Allocating based on profile 0 or max shape.")
                 # Handling dynamic shapes might require more complex allocation
                 # based on Optimization Profiles if not using fixed max size.

            # Calculate buffer size using the determined shape
            size = trt.volume(shape)

            # Allocate pagelocked host memory and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device memory address to the bindings list
            bindings.append(int(device_mem))

            # Append buffer info to inputs or outputs list
            buffer_info = {'host': host_mem, 'device': device_mem, 'shape': shape, 'dtype': dtype}
            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(buffer_info)
            else:
                outputs.append(buffer_info)

        return inputs, outputs, bindings, stream

    def infer(self, input_image):
        """
        Performs inference on a single preprocessed input image.
        input_image: numpy array (CHW format, normalized)
        """
        input_binding = self.inputs[0] # Assigns the first input buffer info
        expected_shape_chw = tuple(input_binding['shape'][1:])

        # Add batch dimension if model expects (N,C,H,W) and input is (C,H,W)
        if len(input_binding['shape']) == 4 and input_binding['shape'][0] == 1:
             if input_image.shape == expected_shape_chw:
                  input_image_batch = np.expand_dims(input_image, axis=0)
             elif input_image.shape == tuple(input_binding['shape']):
                  input_image_batch = input_image # Already has batch dim
             else:
                  raise ValueError(f"Input image shape {input_image.shape} does not match expected CHW shape {expected_shape_chw} or NCHW shape {input_binding['shape']}")
        # Handle cases where model might expect CHW directly (less common with explicit batch)
        elif input_image.shape == tuple(input_binding['shape']):
             input_image_batch = input_image # Input matches model expectation directly
        else:
             raise ValueError(f"Input image shape {input_image.shape} does not match model input shape {input_binding['shape']}")


        # Copy input data to host buffer
        np.copyto(input_binding['host'], input_image_batch.ravel())

        # Transfer input data from host to device (GPU) asynchronously.
        cuda.memcpy_htod_async(input_binding['device'], input_binding['host'], self.stream)

        # --- WORKAROUND: Use synchronous execute_v2 ---
        # As execute_async/execute_async_v2 are missing in this environment setup
        self.context.execute_v2(bindings=self.bindings)
        # --------------------------------------------

        # Transfer predictions from device (GPU) back to host asynchronously.
        # Note: Although execute_v2 is sync, the copies can still be async.
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # Synchronize the stream to ensure MEMORY COPIES are complete.
        # While execute_v2 blocks, the dto copy might still be in flight.
        self.stream.synchronize()
        # ------------------------------------------------------

        # Return the host buffers containing the inference results.
        return [out['host'].reshape(out['shape']) for out in self.outputs]


# --- Main Execution ---

if __name__ == '__main__':
    print("Loading TensorRT engine...")
    # --- Configuration ---
    # engine_path = 'D:/Code/DNN/yolov5/yolov5s.trt'  # Absolute path to the engine file
    engine_path = './yolov5s.trt'
    video_path = 1  # Use 1 for your iPhone/Elgato feed, 0 for default webcam, or a video file path
    input_shape_wh = (640, 640)  # Input shape (width, height) used during engine creation

    try:
        trt_inference = TRTInference(engine_path)
        print("TensorRT engine loaded successfully.")
    except Exception as e:
        print(f"Failed to initialize TensorRT Inference: {e}")
        # Print full traceback for initialization errors
        import traceback
        traceback.print_exc()
        exit()

    # --- Video Capture ---
    print(f"Opening video source: {video_path}")
    # Use DirectShow backend explicitly if needed, might help with some cameras on Windows
    # cap = cv2.VideoCapture(video_path, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}.")
        exit()

    print("Starting inference loop... Press 'q' to quit.")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Info: End of video stream or cannot read frame.")
            break # End of video or error reading frame

        frame_count += 1
        # --- Preprocessing ---
        original_shape_hw = frame.shape[:2]  # (height, width)
        input_data = preprocess_image(frame, input_shape_wh)

        # --- Inference ---
        try:
            outputs = trt_inference.infer(input_data)
            # Assuming the first output is the main detection output for YOLOv5
            raw_output = outputs[0]
        except Exception as e:
            print(f"\n--- Inference Error on frame {frame_count} ---")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Details: {e}")
            # Print full traceback for inference errors
            import traceback
            traceback.print_exc()
            # Optionally continue to next frame or break
            # continue
            break

        # --- Postprocessing ---
        # Verify this shape based on your model. Print raw_output.shape if unsure.
        # print("Raw output shape:", raw_output.shape) # Example: (1, 25200, 85)
        detections = postprocess_output(raw_output, input_shape_wh, original_shape_hw)

        # --- Visualization ---
        for box, score, class_id in detections:
            left, top, width, height = box
            color = (0, 255, 0) # Green for bounding boxes

            # Get class name if available
            if CLASS_NAMES and 0 <= class_id < len(CLASS_NAMES):
                label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
            else:
                label = f"Class {class_id}: {score:.2f}" # Fallback label

            # Draw bounding box
            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 2)

            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Adjust label position to prevent going off-screen top
            label_top = max(top, label_height + 10)
            cv2.rectangle(frame, (left, label_top - label_height - 10), (left + label_width, label_top - baseline), color, cv2.FILLED)

            # Draw label text
            cv2.putText(frame, label, (left, label_top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Black text

        # --- Display Frame ---
        cv2.imshow('YOLOv5 TensorRT Inference', frame)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit requested.")
            break

    # --- Cleanup ---
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    # Explicitly delete TRT objects if needed, though Python's GC and pycuda.autoinit handle much of it
    # del trt_inference # Uncomment if you suspect resource leaks

    print("Script finished.")