def read_annotations_file(file_path):
    video_caption_count = {}
    with open(file_path, 'r') as file:
        for line in file:
            video_name, caption = line.strip().split(' ', 1)
            if video_name in video_caption_count:
                video_caption_count[video_name] += 1
            else:
                video_caption_count[video_name] = 1
    return video_caption_count


annotations_file_path = '../../data/Captions.txt'
video_caption_count = read_annotations_file(annotations_file_path)

caption_amts = set()

for video_name, caption_count in video_caption_count.items():
    print(f"Video Name: {video_name}, Caption Count: {caption_count}")
    caption_amts.add(caption_count)

print("Different amounts of captions for a given video:", caption_amts)