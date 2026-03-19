from datasets import load_dataset
import time

if __name__ == "__main__":
    # Hugging Face will find all .tar archives in the folder and collect them into a dataset
    dataset = load_dataset(
        "webdataset", 
        data_dir="/home/nikita/balalaika/balalaika_data_webdataset", 
        split="train",
        streaming=True # Streaming reading, does not fill up the RAM
    )

    for item in dataset:
        print(f"=== Sample: {item['__key__']} ===")
        
        # Audio will be automatically loaded as a dictionary (NumPy array and Sampling Rate)
        # Hugging Face automatically names the columns by the extensions from the archive
        audio_key = next((k for k in item.keys() if k in ['mp3', 'wav', 'flac', 'ogg']), None)
        
        if audio_key:
            audio_data = item[audio_key]
            print(f"Audio Array: {audio_data['array'].shape}, SR: {audio_data['sampling_rate']}")
        
        # JSON will be automatically parsed
        print(item['json'])
        
        print("-" * 50)
        time.sleep(1)