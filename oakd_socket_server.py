import depthai as dai
import cv2
import socket
import struct
import numpy as np
import time
import os

# Path to the UNIX socket file:
SOCKET_FILE = "/tmp/oakd_socket"

def create_pipeline():
    pipeline = dai.Pipeline()

    # Mono cameras
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()

    # Instead of LEFT/RIGHT, you can use CAM_B/CAM_C.
    # This is purely to remove the deprecation warnings.
    # If you prefer to keep LEFT/RIGHT, you can, but you'll see warnings.
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.createStereoDepth()
    # Instead of setConfidenceThreshold(...)
    stereo.initialConfig.setConfidenceThreshold(230)
    stereo.setLeftRightCheck(True)
    # Use the new median filter name:
    stereo.setMedianFilter(dai.MedianFilter.KERNEL_3x3)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    # Color camera
    colorCam = pipeline.createColorCamera()
    colorCam.setBoardSocket(dai.CameraBoardSocket.RGB)
    colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    colorCam.setFps(30)
    colorCam.setInterleaved(False)

    xoutColor = pipeline.createXLinkOut()
    xoutColor.setStreamName("color")
    colorCam.video.link(xoutColor.input)

    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    return pipeline


def main():
    # Clean up old socket file if it exists:
    if os.path.exists(SOCKET_FILE):
        os.remove(SOCKET_FILE)

    # Create a socket in the file system namespace
    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(SOCKET_FILE)
    server_socket.listen(1)
    print(f"[SERVER] Listening on UNIX socket: {SOCKET_FILE}")

    # Connect to the OAK-D
    pipeline = create_pipeline()
    with dai.Device(pipeline) as device:
        qColor = device.getOutputQueue(name="color", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        print("[SERVER] Waiting for a client to connect...")
        client_conn, _ = server_socket.accept()
        print("[SERVER] Client connected!")

        try:
            while True:
                # 1) Get frames from OAK-D
                inColor = qColor.tryGet()
                inDepth = qDepth.tryGet()

                if inColor is None or inDepth is None:
                    time.sleep(0.01)
                    continue

                colorFrame = inColor.getCvFrame()    # shape: (H, W, 3), uint8
                depthFrame = inDepth.getFrame()      # shape: (H, W), uint16 by default

                # Optional: convert depth to 8-bit to keep example smaller.
                # Real usage might want 16-bit raw, or a disparity map, or something else
                depthFrame8 = (depthFrame / 256).astype(np.uint8)

                # 2) Encode frames so they can be sent over socket
                #    We’ll do JPEG for the color frame. For the depth, we’ll use PNG
                #    so we preserve 8-bit values well. 
                _, colorEnc = cv2.imencode(".jpg", colorFrame)
                _, depthEnc = cv2.imencode(".png", depthFrame8)

                colorBytes = colorEnc.tobytes()
                depthBytes = depthEnc.tobytes()

                # 3) Send length + data for color, then length + data for depth
                #    We’ll send:
                #      [ 4-byte length of color data ][ color data ]
                #      [ 4-byte length of depth data ][ depth data ]
                #    If you prefer, you can store them in one bundle. This is just an example.
                hdr = struct.pack("!II", len(colorBytes), len(depthBytes))
                client_conn.sendall(hdr)
                client_conn.sendall(colorBytes)
                client_conn.sendall(depthBytes)

        except (BrokenPipeError, ConnectionResetError):
            print("[SERVER] Client disconnected.")
        finally:
            client_conn.close()
            server_socket.close()
            print("[SERVER] Socket closed.")

if __name__ == "__main__":
    main()
