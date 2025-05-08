from flask import Flask, render_template, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')  
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, request, render_template

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import io
import base64

from src.utils.video_utils import extract_pose
from src.models.posegcn import PoseGCN


class CustomViolenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(34, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, 2)
    def forward(self, x):
        batch_size, seq_len, joints, coords = x.shape
        x = x.reshape(batch_size, seq_len, joints * coords)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'MP4'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(model_path):
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, torch.nn.Module):
        return state_dict
    if isinstance(state_dict, dict):
        model = CustomViolenceModel()
        if any('2.lstm' in k for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('2.lstm'):
                    new_key = k.replace('2.lstm', 'lstm')
                    new_state_dict[new_key] = v
                elif k.startswith('2.fc'):
                    new_key = k.replace('2.fc', 'fc')
                    new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        return model
    raise ValueError("Unrecognized model format")
model = load_model('/Users/girishkumarpatchikoru/Desktop/appflask/checkpoints/bestgk.pth').to(device)
model.eval()
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def analyze_video(video_path):
    seq_len = 16
    skip = 1
    overlap = 0.75
    threshold = 0.65  # Increased from 0.55
    sports_threshold = 0.80  # Higher threshold for sports content
    MIN_PROB_THRESHOLD = 0.1
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}
    frames = []
    idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    
    if len(frames) < seq_len:
        return {"error": f"Not enough frames ({len(frames)}). Need at least {seq_len}."}
    
    # Extract poses for all frames first
    all_poses = []
    for frame in frames:
        pose = extract_pose(frame)
        all_poses.append(pose)
    all_poses = np.array(all_poses)
    
    # Now we can detect if this is a sports video
    is_sports_video = detect_sports_activity(all_poses)
    
    results = []
    frame_movements = []
    step = max(1, int(seq_len * (1 - overlap))) 
    for i in range(0, len(frames) - seq_len + 1, step):
        window = frames[i:i+seq_len]
        
        poses = np.stack([extract_pose(f) for f in window], axis=0)
        poses_tensor = torch.FloatTensor(poses).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(poses_tensor)
            probs = torch.softmax(output, dim=1)
            violence_prob = probs[0, 1].item()      
        frame_idx = i * skip
        timestamp = frame_idx / fps
        results.append((timestamp, violence_prob))
    
    # Choose appropriate threshold
    effective_threshold = sports_threshold if is_sports_video else threshold
    
    # Calculate metrics with effective threshold
    avg_prob = sum(prob for _, prob in results) / len(results)
    violence_segments = sum(1 for _, prob in results if prob > effective_threshold)
    violence_ratio = violence_segments / len(results) if results else 0
    max_prob = max(prob for _, prob in results)
    min_prob = min(prob for _, prob in results)
    prob_spread = max_prob - min_prob
    
    # Calculate temporal consistency
    consecutive_required = 3 if is_sports_video else 2
    has_consecutive = False
    current_streak = 0
    max_consecutive = 0
    
    for _, prob in results:
        if prob > effective_threshold:
            current_streak += 1
            max_consecutive = max(max_consecutive, current_streak)
            if current_streak >= consecutive_required:
                has_consecutive = True
        else:
            current_streak = 0
    
    # Use better decision logic
    is_violent = has_consecutive and (
        (is_sports_video and violence_ratio > 0.25 and max_prob > sports_threshold) or
        (not is_sports_video and violence_ratio > 0.15 and max_prob > threshold)
    )
    
    # Create visualization
    timestamps = [t for t, _ in results]
    probabilities = [p for _, p in results]
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, probabilities, 'r-o', linewidth=2, markersize=4)
    plt.axhline(y=effective_threshold, color='b', linestyle='--', alpha=0.7)
    plt.fill_between(timestamps, probabilities, effective_threshold, 
                     where=[p > effective_threshold for p in probabilities], 
                     color='r', alpha=0.3, interpolate=True)
    plt.fill_between(timestamps, probabilities, effective_threshold, 
                     where=[p <= effective_threshold for p in probabilities], 
                     color='g', alpha=0.3, interpolate=True)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Violence Probability')
    plt.title('Violence Detection Results')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close()
    
    return {
        "is_violent": is_violent,
        "is_sports": is_sports_video,
        "effective_threshold": effective_threshold,
        "avg_prob": avg_prob,
        "max_prob": max_prob,
        "prob_spread": prob_spread,
        "violence_segments": violence_segments,
        "total_segments": len(results),
        "violence_ratio": violence_ratio,
        "max_consecutive": max_consecutive,
        "plot_img": img_b64
    }
def detect_sports_activity(poses):
    """
    Detect if the video contains sports activities based on pose patterns.
    
    Args:
        poses: Numpy array of pose keypoints
        
    Returns:
        bool: True if sports activity is detected
    """
    # Check for regular, rhythmic movements
    frame_to_frame_changes = []
    for i in range(1, len(poses)):
        change = np.mean(np.abs(poses[i] - poses[i-1]))
        frame_to_frame_changes.append(change)
    
    if len(frame_to_frame_changes) < 5:
        return False
    
    # Regular movements have more consistent frame-to-frame changes
    changes_std = np.std(frame_to_frame_changes)
    
    # Check for high overall motion
    overall_motion = np.mean(frame_to_frame_changes)
    
    # Check for symmetrical movements (common in sports)
    symmetry_score = check_movement_symmetry(poses)
    
    # Combined heuristic
    is_sports = (
        changes_std < 0.05 and  # Consistent movements
        overall_motion > 0.1 and  # High activity
        symmetry_score > 0.6      # Symmetrical movements
    )
    
    return is_sports

def check_movement_symmetry(poses):
    """Helper function to check for symmetrical movements"""
    # Simple implementation - analyzing left/right symmetry
    symmetry_score = 0.7  # Default score
    
    # Check left/right arm and leg symmetry
    # Left-right joint pairs: [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
    limb_pairs = [(5,6), (7,8), (9,10), (13,14), (15,16)]
    
    symmetry_values = []
    for left, right in limb_pairs:
        if left < poses.shape[1] and right < poses.shape[1]:
            left_movements = np.diff(poses[:, left, :], axis=0)
            right_movements = np.diff(poses[:, right, :], axis=0)
            
            if len(left_movements) > 0 and len(right_movements) > 0:
                left_mag = np.sum(np.abs(left_movements))
                right_mag = np.sum(np.abs(right_movements))
                
                if left_mag + right_mag > 0:
                    # Calculate symmetry as the ratio of the smaller to larger magnitude
                    ratio = min(left_mag, right_mag) / max(left_mag, right_mag)
                    symmetry_values.append(ratio)
    
    if symmetry_values:
        symmetry_score = np.mean(symmetry_values)
    
    return symmetry_score
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return redirect(request.url)
    file = request.files['video']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        results = analyze_video(filepath)
        
        if 'error' in results:
            return render_template('error.html', error=results['error'])
        
        return render_template('results.html', results=results, video_name=filename)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
