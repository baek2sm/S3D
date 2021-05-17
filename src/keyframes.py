import numpy as np
import cv2


def get_optical_flow_vectors_from_adjacent_frames(frame0, frame1, quality_level=0.001, min_distance=3, block_size=2):    
    # get adjacent frames
    frame0 = np.array(frame0, dtype=np.uint8)
    frame1 = np.array(frame1, dtype=np.uint8)

    # Shi-Tomasi corner detection for tracking features
    p0 = cv2.goodFeaturesToTrack(frame0,
                                 mask=None,
                                 maxCorners=100,
                                 qualityLevel=quality_level,
                                 minDistance=min_distance,
                                 blockSize=block_size)

    # Lucas-Kanade tracker for extracting optical flow vectors
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame0,
                                           frame1,
                                           p0,
                                           None,
                                           winSize=(3, 3),
                                           maxLevel=0,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # return optical flow vectors
    prior_point = p0[st == 1]
    later_point = p1[st == 1]
    return prior_point, later_point    


def get_key_frames_from_videos(videos, y, length=16):
    
    length_of_videos = len(y)    
    videos_with_keyframes = []
    targets = []
    
    # extracting key frames from videos
    for idx, video in enumerate(videos):
        print('processing {} of {}'.format(idx, length_of_videos))
        
        # handling exception
        if len(video) < length+1:
            print('length of video is too short.')
            continue
        
        frame_and_distance = []
        for frame_idx in range(1, len(video)):
            # get optical flow vectors
            opt_vectors = get_optical_flow_vectors_from_adjacent_frames(video[frame_idx], video[frame_idx-1])
            
            # calculate Manhattan distance of optical flow vectors
            distance = np.sum(np.abs(opt_vectors[0]-opt_vectors[1]))
            
            # save frame index and distance
            frame_and_distance.append((frame_idx, distance))
            
        # sorting and slicing to get key frames
        key_frames_idx = sorted(frame_and_distance, key=lambda x: x[1], reverse=True)[:length]        
        # sorting key frames chronologically
        key_frames_idx =[i[0] for i in sorted(key_frames_idx, key=lambda x: x[0])]
        # get key frames
        video_with_key_frame = np.array(video)[key_frames_idx]        
        
        # add key frames
        videos_with_keyframes.append(video_with_key_frame)
        targets.append(y[idx])
    
    # return key frames
    return np.array(videos_with_keyframes), np.array(targets)