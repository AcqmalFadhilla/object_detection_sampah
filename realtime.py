import cv2
import torch
from torchvision import transforms
from utils import *
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'model/checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint, map_location=torch.device('cpu')) 
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect_video():
    cap = cv2.VideoCapture(0)  # Menggunakan webcam, ganti 0 dengan alamat video jika menggunakan file video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Transform
        image = normalize(to_tensor(resize(frame_pil)))

        # Move to default device
        image = image.to(device)

        # Forward prop.
        predicted_locs, predicted_scores = model(image.unsqueeze(0))

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=0.2,
                                                                 max_overlap=0.5, top_k=200)

        # Move detections to the CPU
        det_boxes = det_boxes[0].to('cpu')
        det_scores = det_scores[0].to('cpu')

        # Transform to original image dimensions
        original_dims = torch.FloatTensor([frame_pil.width, frame_pil.height, frame_pil.width, frame_pil.height]).unsqueeze(0)
        det_boxes = det_boxes * original_dims

        # Decode class integer labels
        det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

        # Annotate
        for i in range(det_boxes.size(0)):
            box_location = det_boxes[i].tolist()
            class_label = det_labels[i].upper()
            confidence = det_scores[i].item()
            label_text = f"{class_label}: {confidence:.2f}"
            if confidence > 0.6:
                cv2.rectangle(frame, (int(box_location[0]), int(box_location[1])), (int(box_location[2]), int(box_location[3])), 10)
                cv2.putText(frame, label_text, (int(box_location[0]), int(box_location[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 10)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_video()
