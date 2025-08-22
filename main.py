import cv2
import numpy as np
import mss
import keyboard
import onnxruntime as ort
import time
import ctypes
import threading
import queue

# --- CONFIGURATION ---
CONFIG = {
    # Screen Capture
    "capture_size": 900,
    "monitor": 2,
    
    # Model & Detection
    "onnx_model_path": "best.onnx",
    "conf_threshold": 0.5,

    # Aiming & Control
    "toggle_key": "f12",
    "quit_key": "f11",

    # Fine-tuning
    "detection_offset_x": 40,
    "detection_offset_y": 60,

    # Resolution-Independent PID Controller
    "pid_kp": 0.2,   # Proportional: The main magnet strength.
    "pid_ki": 0.029,  # Integral: Corrects for lag on moving targets.
    "pid_kd": 0.015, # Derivative: Smooths movement and prevents overshoot.
    "sensitivity": 85.0, # Master sensitivity. Scales the PID output to mouse movement.
}
# --- END CONFIGURATION ---

class PID:
    """A robust, time-aware PID controller."""
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

    def update(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if dt == 0:
            return 0

        self.integral += error * dt
        self.integral = max(min(self.integral, 1.0), -1.0) # Clamp integral

        derivative = (error - self.last_error) / dt
        self.last_error = error
        
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def reset(self):
        self.last_error = 0
        self.integral = 0
        self.last_time = time.time()

class ScreenCapturer:
    def __init__(self, config, frame_queue):
        self.config = config
        self.frame_queue = frame_queue
        self.running = True
        with mss.mss() as sct:
            self.monitor = self._get_capture_monitor(sct, config["capture_size"])
            self.width = self.monitor['width']
            self.height = self.monitor['height']

    def _get_capture_monitor(self, sct, size):
        try:
            target_monitor = sct.monitors[self.config["monitor"]]
        except IndexError:
            print(f"[ERROR] Monitor {self.config['monitor']} not found. Using primary monitor.")
            target_monitor = sct.monitors[1]
        
        center_x = target_monitor["left"] + target_monitor["width"] // 2
        center_y = target_monitor["top"] + target_monitor["height"] // 2
        left = center_x - (size // 2)
        top = center_y - (size // 2)
        return {"top": top, "left": left, "width": size, "height": size}

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        with mss.mss() as sct:
            while self.running:
                screenshot = sct.grab(self.monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    time.sleep(0.001)

    def stop(self):
        self.running = False

class Detector:
    def __init__(self, config, frame_queue, results_queue):
        self.config = config
        self.frame_queue = frame_queue
        self.results_queue = results_queue
        self.running = True
        try:
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(config["onnx_model_path"], providers=providers)
            print(f"[INFO] ONNX model loaded with provider: {self.session.get_providers()[0]}")
        except Exception as e:
            print(f"[ERROR] Could not load ONNX model: {e}")
            exit()
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_height, self.input_width = model_inputs[0].shape[2], model_inputs[0].shape[3]

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                target = self.detect(frame)
                if self.results_queue.full():
                    self.results_queue.get_nowait()
                self.results_queue.put(target)
            except queue.Empty:
                continue

    def stop(self):
        self.running = False

    def detect(self, frame):
        input_tensor, scale, pad_lt = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self._postprocess(outputs[0], scale, pad_lt)

    def _preprocess(self, img):
        h, w, _ = img.shape
        scale = min(self.input_width / w, self.input_height / h)
        resized_w, resized_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = self.input_width - resized_w, self.input_height - resized_h
        top, left = pad_y // 2, pad_x // 2
        padded_img = cv2.copyMakeBorder(img_resized, top, pad_y - top, left, pad_x - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        padded_img = padded_img.transpose(2, 0, 1)
        return np.expand_dims(padded_img, axis=0), scale, (left, top)

    def _postprocess(self, output, scale, pad_lt):
        detections = output[0]
        pad_l, pad_t = pad_lt
        best_confidence, best_target = 0, None
        for pred in detections:
            confidence = pred[4]
            if confidence > self.config["conf_threshold"] and confidence > best_confidence:
                if int(pred[5]) == 0:
                    best_confidence = confidence
                    cx_padded, cy_padded = pred[0], pred[1]
                    cx_unpadded, cy_unpadded = cx_padded - pad_l, cy_padded - pad_t
                    best_target = (int(cx_unpadded / scale), int(cy_unpadded / scale))
        return best_target

class Aimbot:
    def __init__(self, config):
        self.config = config
        self.aim_enabled = False
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        self.results_queue = queue.Queue(maxsize=1)
        self.capturer = ScreenCapturer(config, self.frame_queue)
        self.detector = Detector(config, self.frame_queue, self.results_queue)
        self.capture_cx = self.capturer.width // 2
        self.capture_cy = self.capturer.height // 2
        
        self.pid_x = PID(config["pid_kp"], config["pid_ki"], config["pid_kd"])
        self.pid_y = PID(config["pid_kp"], config["pid_ki"], config["pid_kd"])

    def _handle_keys(self):
        if keyboard.is_pressed(self.config["toggle_key"]):
            self.aim_enabled = not self.aim_enabled
            print(f"[INFO] Aim Assist {'ENABLED' if self.aim_enabled else 'DISABLED'}")
            self.pid_x.reset()
            self.pid_y.reset()
            time.sleep(0.3)
        if keyboard.is_pressed(self.config["quit_key"]):
            self.running = False

    def move_mouse_to_target(self, target_cx, target_cy):

        error_x = (target_cx - self.capture_cx) / (self.capturer.width / 2.0)
        error_y = (target_cy - self.capture_cy) / (self.capturer.height / 2.0)

        correction_x = self.pid_x.update(error_x)
        correction_y = self.pid_y.update(error_y)

        move_x = correction_x * self.config["sensitivity"]
        move_y = correction_y * self.config["sensitivity"]

        ctypes.windll.user32.mouse_event(0x0001, int(move_x), int(move_y), 0, 0)

    def run(self):
        print(f"[INFO] Press '{self.config['toggle_key'].upper()}' to toggle aim assist.")
        print(f"[INFO] Press '{self.config['quit_key'].upper()}' to quit.")
        self.capturer.start()
        self.detector.start()

        window_name = "AI Aim Assist"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.capturer.width, self.capturer.height)
        cv2.moveWindow(window_name, 100, 100)

        target = None
        start_time, frame_count, fps = time.time(), 0, 0

        try:
            frame = self.frame_queue.get(timeout=2)
        except queue.Empty:
            print("[ERROR] Could not get initial frame.")
            self.running = False
            frame = np.zeros((self.capturer.height, self.capturer.width, 3), dtype=np.uint8)

        while self.running:
            self._handle_keys()
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                target = self.results_queue.get_nowait()
            except queue.Empty:
                pass

            if self.aim_enabled and target is not None:
                offset_cx = target[0] + self.config["detection_offset_x"]
                offset_cy = target[1] + self.config["detection_offset_y"]
                
                self.move_mouse_to_target(offset_cx, offset_cy)
                
                cv2.circle(frame, (offset_cx, offset_cy), 5, (0, 0, 255), -1)
            else:
                self.pid_x.reset()
                self.pid_y.reset()

            frame_count += 1
            if time.time() - start_time >= 1:
                fps = frame_count
                frame_count = 0
                start_time = time.time()
            
            cv2.putText(frame, f"Display FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)

            if cv2.waitKey(1) == ord("q"): self.running = False
        
        print("[INFO] Shutting down...")
        self.capturer.stop()
        self.detector.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    aimbot = Aimbot(CONFIG)
    aimbot.run()