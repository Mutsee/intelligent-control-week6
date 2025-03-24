from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("rail-segmentation/best.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)  # Menampilkan hasil
    
    # **Menunggu input 'q' untuk keluar**
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Jika tombol 'q' ditekan
            break

    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV

# Contoh penggunaan
detect_rail_lane("dataset/sample_image.jpg")
