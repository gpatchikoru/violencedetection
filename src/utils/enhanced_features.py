import numpy as np

def compute_enhanced_features(poses):
    seq_len, n_joints, coords = poses.shape
    
    features = []
    raw_features = poses.reshape(seq_len, -1)  
    features.append(raw_features)
    velocity = np.zeros_like(poses)
    velocity[1:] = poses[1:] - poses[:-1]
    vel_features = velocity.reshape(seq_len, -1)
    features.append(vel_features)
    accel = np.zeros_like(poses)
    accel[1:] = velocity[1:] - velocity[:-1]
    accel_features = accel.reshape(seq_len, -1)
    features.append(accel_features)
    
    center = poses[:, 0:1, :]  
    distances = poses - center  
    dist_features = distances.reshape(seq_len, -1)
    features.append(dist_features)
    
    
    joint_pairs = [
        (5, 6),   
        (9, 10),  
        (13, 14), 
        (15, 16)  
    ]
    
    pair_distances = []
    for j1, j2 in joint_pairs:
        dist = np.linalg.norm(poses[:, j1, :] - poses[:, j2, :], axis=1)
        pair_distances.append(dist)
    
    pair_features = np.stack(pair_distances, axis=1)  
    features.append(pair_features)
    
   
    combined = np.concatenate(features, axis=1)
    return combined

def augment_pose_sequence(poses, augment_prob=0.5):
    augmented = poses.copy()
    
    
    if np.random.random() < augment_prob:
        noise_level = np.random.uniform(0.005, 0.015)
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented += noise
    
    
    if np.random.random() < augment_prob:
       
        pairs = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
        for left, right in pairs:
            augmented[:, [left, right]] = augmented[:, [right, left]].copy()
    
   
    if np.random.random() < augment_prob:
        seq_len = poses.shape[0]
        
        new_len = int(seq_len * np.random.uniform(0.8, 1.2))
        new_len = max(seq_len // 2, min(seq_len * 2, new_len))
        
      
        indices = np.linspace(0, seq_len - 1, new_len)
        warped = np.zeros((new_len, poses.shape[1], poses.shape[2]))
        
        for i, idx in enumerate(indices):
          
            idx_floor = int(np.floor(idx))
            idx_ceil = min(idx_floor + 1, seq_len - 1)
            alpha = idx - idx_floor
            warped[i] = (1 - alpha) * augmented[idx_floor] + alpha * augmented[idx_ceil]
        
       
        indices = np.linspace(0, new_len - 1, seq_len)
        resampled = np.zeros_like(augmented)
        
        for i, idx in enumerate(indices):
            idx_floor = int(np.floor(idx))
            idx_ceil = min(idx_floor + 1, new_len - 1)
            alpha = idx - idx_floor
            resampled[i] = (1 - alpha) * warped[idx_floor] + alpha * warped[idx_ceil]
        
        augmented = resampled
    
    return augmented