import os
import cv2
import numpy as np
import pickle

# CONSTANTS
frames = 5 # (Maximum 10)
captions = 5 # (Maximum 18)
video_dir = '../data/YouTubeClips/'
caption_file = '../data/Captions.txt'
output_dir = f'../data/preprocessed_{frames}frames_{captions}captions/'

#--------------------

class ClipsCaptions:
    def __init__(self, video_name, captions, frames):
        self.video_name = video_name
        self.captions = captions
        self.frames = frames

    @staticmethod
    def extract_frames(video_path, num_frames):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate step size to sample frames evenly
        step_size = total_frames // num_frames
        frame_indices = [i * step_size for i in range(num_frames)]
        
        frames = []
        for index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Resize frame to match ConvLSTM2D input shape
            frames.append(frame)
        
        cap.release()
        if(len(frames) < num_frames):
            print("\n\n", "Warning: Less frames than expected: ", len(frames), "\n\n")
        return np.array(frames)


    @classmethod
    def from_video(cls, video_path, caption_file, num_frames=10, num_captions=5):
        # Extract frames
        frames = cls.extract_frames(video_path, num_frames)

        # Read captions
        captions = {}
        with open(caption_file, 'r') as f:
            for line in f:
                video_name, caption = line.strip().split(' ', 1)
                captions[video_name] = captions.get(video_name, []) + [caption]

        # Get captions for the video
        video_name = video_path.split('/')[-1].split('.')[0]
        caption_list = captions.get(video_name, [])
        caption_list = caption_list[:num_captions] if len(caption_list) >= num_captions else caption_list + [''] * (num_captions - len(caption_list))

        # Preprocess captions
        for i in range(len(caption_list)):
            caption = caption_list[i]
            # Convert the caption to lowercase
            caption = caption.lower()
            # Remove non-alphabetic characters and split into words
            caption = [word for word in caption.split() if word.isalpha()]
            # Join the words back into a sentence
            caption = ' '.join(caption)
            # Remove additional spaces
            caption = caption.replace('\s+', ' ')
            # Add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word) > 1]) + ' endseq'
            caption_list[i] = caption

        return cls(video_name, caption_list, frames)




if __name__ == "__main__":    
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass # directory exists

    # get list of all .avi files in the directory
    avi_files = [f for f in os.listdir(video_dir) if f.endswith('.avi')]
    curr_count = 1
    for avi_file in avi_files:
        print("Processing video", curr_count, "of", len(avi_files))
        curr_count += 1
        # create ClipsCaptions obj
        video_path = os.path.join(video_dir, avi_file)
        clip_captions = ClipsCaptions.from_video(video_path, caption_file, num_frames=frames, num_captions=captions)
        
        # Serialize instance and save it in the directory
        output_file = os.path.join(output_dir, avi_file.replace('.avi', '.pkl'))
        with open(output_file, 'wb') as f:
            pickle.dump(clip_captions, f)