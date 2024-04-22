import os
import cv2
import matplotlib.pyplot as plt

directory = '../../data/YouTubeClips'

# metrics
frame_rates = set()
min_width, min_height = float('inf'), float('inf')
max_width, max_height = 0, 0
total_width, total_height = 0, 0
total_length = 0
min_length, max_length = float('inf'), 0
num_videos = 0
frame_rate_distribution = {}
length_distribution = {}

# file iteration
for filename in os.listdir(directory):
    if filename.endswith('.avi'):
        filepath = os.path.join(directory, filename)
        video = cv2.VideoCapture(filepath)
        
        # Video properties
        frame_rate = round(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        length = round(num_frames / frame_rate)
        
        # update the extracted info
        frame_rates.add(frame_rate)
        min_width = min(min_width, width)
        min_height = min(min_height, height)
        max_width = max(max_width, width)
        max_height = max(max_height, height)
        total_width += width
        total_height += height
        total_length += length
        min_length = min(min_length, length)
        max_length = max(max_length, length)
        num_videos += 1
        
        # update the distributions
        frame_rate_distribution[frame_rate] = frame_rate_distribution.get(frame_rate, 0) + 1
        length_distribution[length] = length_distribution.get(length, 0) + 1
        
        # close file to save ram
        video.release()



avg_width = total_width / num_videos
avg_height = total_height / num_videos
avg_length = total_length / num_videos


print("Frame Rates:", frame_rates)
print("Minimum Video Dimensions:", min_width, "x", min_height)
print("Maximum Video Dimensions:", max_width, "x", max_height)
print("Average Video Dimensions:", avg_width, "x", avg_height)
print("Minimum Video Length:", min_length, "seconds")
print("Maximum Video Length:", max_length, "seconds")
print("Average Video Length:", avg_length, "seconds")
print("Total Number of Videos:", num_videos)
print("Total Duration:", total_length, "seconds")

# plot distribution of frame rates
plt.bar(frame_rate_distribution.keys(), frame_rate_distribution.values(), color='lightskyblue')
plt.yscale('log')
text_color = 'purple'
for key, value in frame_rate_distribution.items():
    plt.text(key, value, str(value), ha='center', va='bottom', color=text_color)
    text_color = 'darkgreen' if text_color == 'purple' else 'purple'
plt.xlabel('Frame Rate')
plt.ylabel('Frequency')
plt.title('Distribution of Frame Rates')
plt.savefig('frame_rate_distribution.png')
plt.close()

# plot distribution of video lengths
plt.bar(length_distribution.keys(), length_distribution.values(), color='lightskyblue')
plt.xlabel('Video Length (seconds)')
plt.ylabel('Frequency')
plt.title('Video Length Distribution')
plt.savefig('video_length_distribution.png')
plt.close()
