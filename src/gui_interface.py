import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import threading
from interface import *
import youtube_dl

# Import and define the make_average_predictions and download_youtube_videos functions here

def validate_link(link):
    with youtube_dl.YoutubeDL() as ydl:
        try:
            ydl.extract_info(link, download=False)
            return True
        except youtube_dl.DownloadError:
            return False

def select_video():
    clear_results()
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi"), ("All Files", "*.*")])
    if video_path:
        video_entry.delete(0, tk.END)
        video_entry.insert(0, video_path)
        threading.Thread(target=play_video, args=(video_path,)).start()


def process_video():
    video_path = video_entry.get()
    if video_path:
        if os.path.exists(video_path):
            clear_results()
            result_label.config(text="Processing video...")
            window.update()

            threading.Thread(target=process_and_display_results, args=(video_path,)).start()
        else:
            messagebox.showerror("Error", "Video file does not exist.")
    else:
        messagebox.showerror("Error", "Please select a video.")
        
    video_entry.delete(0, tk.END)


def process_and_display_results(video_path):
    results = make_average_predictions(video_path, 50)
    display_results(results)

def display_results(results):
    
    result_label.config(text="Video Processing Completed.")
    window.update()
    result_text.delete('1.0', tk.END)  # Clear previous results
    for result in results:
        result_text.insert(tk.END, result + '\n')


def process_youtube_video():
    output_directory = '../Youtube_Videos'
    os.makedirs(output_directory, exist_ok=True)

    youtube_link = youtube_entry.get()
    if youtube_link:
        if validate_link(youtube_link):
            threading.Thread(target=download_youtube_videos, args=(youtube_link, output_directory)).start()
        else:
            messagebox.showerror("Error", "Invalid YouTube video link.")
    else:
        messagebox.showerror("Error", "Please enter a YouTube video link.")

    youtube_entry.delete(0, tk.END)
    
    
def clear_results():
    result_text.delete('1.0', tk.END)
    window.update()

# Create the main window
window = tk.Tk()
window.title("Video Classification")
window.geometry("800x700")  # Increase the window size

# Video Display Section
video_frame = tk.Frame(window, width=640, height=480)  # Set the desired dimensions for the video display
video_frame.pack(pady=20)

# Selected Video Label
selected_video_label = tk.Label(video_frame, text="Selected Video:")
selected_video_label.pack()

# Video Player
video_player = tk.Label(video_frame, bg="black", relief="solid")
video_player.pack(pady=10)

# Video Selection Section
select_frame = tk.Frame(window)
select_frame.pack(pady=10)

video_label = tk.Label(select_frame, text="Select Video:")
video_label.grid(row=0, column=0, padx=10)

video_entry = tk.Entry(select_frame, width=40)
video_entry.grid(row=0, column=1, padx=10)

select_button = tk.Button(select_frame, text="Browse", command=select_video)
select_button.grid(row=0, column=2, padx=10)

process_button = tk.Button(select_frame, text="Make preditions on Video", command=process_video)
process_button.grid(row=1, column=1, pady=10)

result_label = tk.Label(window, text="")
result_label.pack(pady=10)

result_text = tk.Text(window, height=10, width=70)
result_text.pack(pady=10)

clear_button = tk.Button(window, text="Clear Results", command=clear_results)
clear_button.pack()

# YouTube Link Section
youtube_frame = tk.Frame(window)
youtube_frame.pack(pady=10)

youtube_label = tk.Label(youtube_frame, text="YouTube Link:")
youtube_label.grid(row=0, column=0, padx=10)

youtube_entry = tk.Entry(youtube_frame, width=40)
youtube_entry.grid(row=0, column=1, padx=10)

process_youtube_button = tk.Button(youtube_frame, text="Download YouTube Video", command=process_youtube_video)
process_youtube_button.grid(row=1, column=1, pady=10)

# Scrollable widget to display classes_list
classes_frame = tk.Frame(window)
classes_frame.pack(pady=10)

classes_label = tk.Label(classes_frame, text="Activity List that can be pridicted (you can also choos other set of activites\n by training the model on differen sets o activites):")
classes_label.pack()

classes_listbox = tk.Listbox(classes_frame, height=10, width=70)
classes_listbox.pack(side=tk.LEFT, fill=tk.BOTH)

classes_scrollbar = tk.Scrollbar(classes_frame)
classes_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the scrollbar with the listbox
classes_listbox.config(yscrollcommand=classes_scrollbar.set)
classes_scrollbar.config(command=classes_listbox.yview)

# Populate the classes_listbox with the items from classes_list
for item in classes_list:
    classes_listbox.insert(tk.END, item)
# Start the main event loop


 
 
 
 
#  # Creating The Output directories if it does not exist
# output_directory = '../Youtube_Videos'
# os.makedirs(output_directory, exist_ok = True)
 
# video_link = 'https://www.youtube.com/watch?v=ayI-e3cJM-0'
# # Downloading a YouTube Video
# video_title = download_youtube_videos(video_link ,output_directory)
 
# # Getting the YouTube Video's path you just downloaded
# input_video_file_path = f'{output_directory}/{video_title}.mp4'


# # Setting sthe Window Size which will be used by the Rolling Average Proces
# window_size = 90
 
 
# # Constructing The Output YouTube Video Path
# output_video_file_path = f'{output_directory}/{video_title}.mp4'


 
# # # Calling the predict_on_live_video method to start the Prediction.
# # predict_on_live_video(input_video_file_path, output_video_file_path, window_size)


    
# output_video_file_path = f'{output_directory}/{video_title}.mp4'
# # Calling The Make Average Method To Start The Process
# make_average_predictions(input_video_file_path, 50)
 
# # Play Video File in the Notebook

# play_video(output_video_file_path)